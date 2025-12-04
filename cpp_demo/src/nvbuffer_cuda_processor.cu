#include "nvbuffer_cuda_processor.h"
#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <chrono>
#include <gst/gst.h>
#include <nvbufsurface.h>
#include <nvbufsurftransform.h>
#include <vector>
#include <string>
#include <iostream>

// --- Stitching Structures & Kernel ---

struct float10 {
  float x[10];
};

struct ptr4 {
  uchar4* v[4]; // RGBA
};

static __device__ __forceinline__ uchar4 belend(uchar4 a, uchar4 b, float w) {
  return make_uchar4(
      (unsigned char)(a.x * w + b.x * (1 - w)),
      (unsigned char)(a.y * w + b.y * (1 - w)),
      (unsigned char)(a.z * w + b.z * (1 - w)),
      255
  );
}

static __global__ void surround_kernel(const float10* table, int w, int h,
                                       ptr4 images, int iw, int ih,
                                       uchar4* output, int out_pitch) {
  int ix = blockDim.x * blockIdx.x + threadIdx.x;
  int iy = blockDim.y * blockIdx.y + threadIdx.y;
  if (ix >= w || iy >= h) return;

  // Calculate output address using pitch
  uchar4* out_pixel = (uchar4*)((char*)output + iy * out_pitch) + ix;
  
  int pos = iy * w + ix; // Table is packed
  float10 item = table[pos];
  int flag = (int)item.x[0];
  float weight = item.x[1];

  if (flag == -1) {
      *out_pixel = make_uchar4(0, 0, 0, 255);
      return;
  }
  
  // Inputs are packed (temp buffer)
  if (flag < 4) {
    int x = (int)item.x[2 + flag * 2 + 0];
    int y = (int)item.x[2 + flag * 2 + 1];

    x = max(0, min(x, iw - 1));
    y = max(0, min(y, ih - 1));

    *out_pixel = images.v[flag][y * iw + x];
  } else {
    const int idxs[][2] = {{2, 1}, {0, 3}, {0, 1}, {2, 3}};
    int a = idxs[flag - 4][0];
    int b = idxs[flag - 4][1];
    int ax = (int)item.x[2 + a * 2 + 0];
    int ay = (int)item.x[2 + a * 2 + 1];
    int bx = (int)item.x[2 + b * 2 + 0];
    int by = (int)item.x[2 + b * 2 + 1];
    
    ax = max(0, min(ax, iw - 1));
    ay = max(0, min(ay, ih - 1));
    bx = max(0, min(bx, iw - 1));
    by = max(0, min(by, ih - 1));

    *out_pixel = belend(images.v[a][ay * iw + ax], images.v[b][by * iw + bx], weight);
  }
}

// --- Debug Kernel ---
__global__ void draw_debug_box(uchar4* output, int pitch, int w, int h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < 200 && y < 200 && x < w && y < h) {
        uchar4* pixel = (uchar4*)((char*)output + y * pitch) + x;
        // Red box
        *pixel = make_uchar4(255, 0, 0, 255);
    }
}

// --- Surrounder Class ---

class Surrounder {
 public:
  ~Surrounder() { destroy(); }

  bool load(const std::string& filename, int w, int h, int numcam, int camw, int camh) {
    // Try multiple paths
    std::vector<std::string> paths = {
        filename,
        "../" + filename,
        "../../" + filename,
        "cpp_demo/" + filename,
        "../cpp_demo/" + filename
    };
    
    FILE* f = nullptr;
    std::string loaded_path;
    
    for (const auto& path : paths) {
        f = fopen(path.c_str(), "rb");
        if (f) {
            loaded_path = path;
            break;
        }
    }

    if (f == nullptr) {
      printf("\033[1;31m[Surrounder] ERROR: Failed to find table file: %s\033[0m\n", filename.c_str());
      printf("Searched in:\n");
      for (const auto& path : paths) printf("  - %s\n", path.c_str());
      return false;
    }

    fseek(f, 0, SEEK_END);
    size_t size = ftell(f);
    fseek(f, 0, SEEK_SET);

    size_t expected_size = (size_t)w * h * 10 * sizeof(float);
    if (size != expected_size) {
      printf("\033[1;31m[Surrounder] ERROR: Invalid file size. Expected %lu, got %lu\033[0m\n", expected_size, size);
      fclose(f);
      return false;
    }

    unsigned char* table_host = new unsigned char[size];
    if (fread(table_host, 1, size, f) != size) {
        printf("[Surrounder] Failed to read table file\n");
        delete[] table_host;
        fclose(f);
        return false;
    }
    fclose(f);

    w_ = w;
    h_ = h;
    camw_ = camw;
    camh_ = camh;

    if (cudaMalloc(&table_, size) != cudaSuccess) {
        printf("[Surrounder] cudaMalloc table failed\n");
        delete[] table_host;
        return false;
    }
    
    cudaMemcpy(table_, table_host, size, cudaMemcpyHostToDevice);
    delete[] table_host;
    
    printf("\033[1;32m[Surrounder] Loaded successfully from %s\033[0m\n", loaded_path.c_str());
    return true;
  }

  void process(uchar4* output, int out_pitch, const std::vector<uchar4*>& inputs, cudaStream_t stream = 0) {
      if (inputs.size() != 4) return;
      
      ptr4 images_ptr;
      for(int i=0; i<4; i++) images_ptr.v[i] = inputs[i];
      
      dim3 block(32, 32);
      dim3 grid((w_ + block.x - 1) / block.x, (h_ + block.y - 1) / block.y);
      
      surround_kernel<<<grid, block, 0, stream>>>(
          table_, w_, h_, images_ptr, camw_, camh_, output, out_pitch);
          
      // Draw debug box to prove we touched the frame
      draw_debug_box<<<dim3(10, 10), dim3(20, 20), 0, stream>>>(output, out_pitch, w_, h_);
  }

 private:
  void destroy() {
    if (table_) {
      cudaFree(table_);
      table_ = nullptr;
    }
  }

  float10* table_ = nullptr;
  int w_ = 0;
  int h_ = 0;
  int camw_ = 0;
  int camh_ = 0;
};

// --- Global State ---

static Surrounder* g_surrounder = nullptr;
static uchar4* g_temp_input = nullptr; // Temporary buffer to hold input frame (packed)
static int process_count = 0; // Moved to global scope for debug printing

// --- NvBuffer Processor Implementation ---

bool cuda_init() {
    int devCount;
    if (cudaGetDeviceCount(&devCount) != cudaSuccess) return false;
    return devCount > 0;
}

void cuda_cleanup() {
    if (g_surrounder) {
        delete g_surrounder;
        g_surrounder = nullptr;
    }
    if (g_temp_input) {
        cudaFree(g_temp_input);
        g_temp_input = nullptr;
    }
}

bool nvbuffer_cuda_process(GstBuffer *buffer, int width, int height, int brighten_value) {
    static bool first_run = true;
    static bool stitching_enabled = false;
    process_count++;
    
    // Initialize Surrounder on first run
    if (first_run) {
        g_surrounder = new Surrounder();
        // Load the binary table. Assuming file is in current directory.
        // Input/Output dimensions match the pipeline resolution (1920x1080)
        if (g_surrounder->load("surround_view.binary", width, height, 4, width, height)) {
            stitching_enabled = true;
            
            // Allocate temp buffer (packed RGBA)
            int size = width * height * sizeof(uchar4);
            if (cudaMalloc(&g_temp_input, size) != cudaSuccess) {
                printf("Failed to allocate temp buffer. Stitching disabled.\n");
                stitching_enabled = false;
            }
        } else {
            printf("Stitching disabled (Table load failed).\n");
        }
        first_run = false;
    }

    // Map Buffer
    GstMapInfo map_info;
    if (!gst_buffer_map(buffer, &map_info, (GstMapFlags)(GST_MAP_READ | GST_MAP_WRITE))) {
        return false;
    }
    
    NvBufSurface *surf = (NvBufSurface *)map_info.data;
    
    // We need to map the surface to access dataPtr if it's not already mapped? 
    // Actually dataPtr is usually valid for CUDA on Jetson without Map, 
    // but NvBufSurfaceMap ensures synchronization and validity.
    // Use NVBUF_MAP_READ_WRITE just in case, though we primarily use dataPtr.
    if (!surf || NvBufSurfaceMap(surf, 0, -1, NVBUF_MAP_READ_WRITE) != 0) {
        gst_buffer_unmap(buffer, &map_info);
        return false;
    }
    
    // Sync for device (ensure GPU sees latest data)
    NvBufSurfaceSyncForDevice(surf, 0, -1);
    
    NvBufSurfaceParams *params = &surf->surfaceList[0];
    uchar4* d_surface = (uchar4*)params->dataPtr;
    int pitch = params->pitch;
    
    if (first_run || process_count % 30 == 0) {
        printf("[Debug] Surface Layout: %d (0=Pitch, 1=BlockLinear), Pitch: %d, Width: %d, Height: %d\n", 
               params->layout, pitch, params->width, params->height);
    }

    if (!d_surface) {
        NvBufSurfaceUnMap(surf, 0, -1);
        gst_buffer_unmap(buffer, &map_info);
        return false;
    }
    
    if (stitching_enabled && g_temp_input) {
        // 1. Copy input surface (with pitch) to temp buffer (packed)
        cudaError_t err = cudaMemcpy2D(g_temp_input, width * sizeof(uchar4), 
                     d_surface, pitch, 
                     width * sizeof(uchar4), height, 
                     cudaMemcpyDeviceToDevice);
        if (err != cudaSuccess) printf("Memcpy2D Failed: %s\n", cudaGetErrorString(err));
        
        // 2. Prepare inputs (Simulate 4 cameras using the same source)
        std::vector<uchar4*> inputs(4, g_temp_input);
        
        // 3. Run Stitching
        g_surrounder->process(d_surface, pitch, inputs);
        
        err = cudaGetLastError();
        if (err != cudaSuccess) printf("Kernel Launch Failed: %s\n", cudaGetErrorString(err));
        
        cudaDeviceSynchronize();
    }
    
    // Cleanup
    NvBufSurfaceSyncForDevice(surf, 0, -1); // Ensure subsequent elements see changes
    NvBufSurfaceUnMap(surf, 0, -1);
    gst_buffer_unmap(buffer, &map_info);
    
    return true;
}
