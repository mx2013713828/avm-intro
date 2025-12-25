#include "nvbuffer_cuda_processor.h"
#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <vector>
#include <string>

// Forward declare NvBuffer if HAVE_JETSON_NVMM is defined
#ifdef HAVE_JETSON_NVMM
#include <nvbufsurface.h>
#endif

// --- Data Structures ---
struct float10 { float x[10]; };
struct ptr4 { uchar4* v[4]; };

// --- CUDA Kernels ---
static __device__ __forceinline__ uchar4 blend(uchar4 a, uchar4 b, float w) {
    return make_uchar4(
        (unsigned char)(a.x * w + b.x * (1.0f - w)),
        (unsigned char)(a.y * w + b.y * (1.0f - w)),
        (unsigned char)(a.z * w + b.z * (1.0f - w)),
        255
    );
}

static __global__ void surround_kernel(const float10* table, int w, int h,
                                       ptr4 images, int iw, int ih,
                                       uchar4* output, int out_pitch) {
    int ix = blockDim.x * blockIdx.x + threadIdx.x;
    int iy = blockDim.y * blockIdx.y + threadIdx.y;
    if (ix >= w || iy >= h) return;

    uchar4* out_pixel = (uchar4*)((char*)output + iy * out_pitch) + ix;
    float10 item = table[iy * w + ix];
    int flag = (int)item.x[0];
    float weight = item.x[1];

    if (flag == -1) {
        *out_pixel = make_uchar4(0, 0, 0, 255);
        return;
    }

    if (flag < 4) {
        int x = max(0, min((int)item.x[2 + flag * 2 + 0], iw - 1));
        int y = max(0, min((int)item.x[2 + flag * 2 + 1], ih - 1));
        *out_pixel = images.v[flag][y * iw + x];
    } else {
        const int idxs[][2] = {{2, 1}, {0, 3}, {0, 1}, {2, 3}};
        int a = idxs[flag - 4][0];
        int b = idxs[flag - 4][1];
        int ax = max(0, min((int)item.x[2 + a * 2 + 0], iw - 1));
        int ay = max(0, min((int)item.x[2 + a * 2 + 1], ih - 1));
        int bx = max(0, min((int)item.x[2 + b * 2 + 0], iw - 1));
        int by = max(0, min((int)item.x[2 + b * 2 + 1], ih - 1));
        *out_pixel = blend(images.v[a][ay * iw + ax], images.v[b][by * iw + bx], weight);
    }
}

// --- Internal State ---
class Surrounder {
public:
    float10* table = nullptr;
    int w, h, iw, ih;
    
    bool load(const char* filename, int out_w, int out_h, int in_w, int in_h) {
        FILE* f = fopen(filename, "rb");
        if (!f) return false;
        
        size_t size = (size_t)out_w * out_h * sizeof(float10);
        float10* h_table = (float10*)malloc(size);
        if (fread(h_table, 1, size, f) != size) { free(h_table); fclose(f); return false; }
        fclose(f);

        if (table) cudaFree(table);
        cudaMalloc(&table, size);
        cudaMemcpy(table, h_table, size, cudaMemcpyHostToDevice);
        free(h_table);
        
        w = out_w; h = out_h; iw = in_w; ih = in_h;
        return true;
    }
};

static Surrounder* g_surrounder = nullptr;

// --- API Implementation ---
bool cuda_init() { return true; }
void cuda_cleanup() { if (g_surrounder) { if (g_surrounder->table) cudaFree(g_surrounder->table); delete g_surrounder; g_surrounder = nullptr; } }

bool stitching_init(const char* bin_file, int out_w, int out_h, int in_w, int in_h) {
    if (!g_surrounder) g_surrounder = new Surrounder();
    return g_surrounder->load(bin_file, out_w, out_h, in_w, in_h);
}

bool stitching_process(uchar4* out_ptr, int out_pitch, const std::vector<uchar4*>& in_ptrs) {
    if (!g_surrounder || !g_surrounder->table || in_ptrs.size() != 4) return false;
    
    ptr4 images;
    for (int i = 0; i < 4; i++) images.v[i] = in_ptrs[i];
    
    dim3 block(32, 32);
    dim3 grid((g_surrounder->w + block.x - 1) / block.x, (g_surrounder->h + block.y - 1) / block.y);
    surround_kernel<<<grid, block>>>(g_surrounder->table, g_surrounder->w, g_surrounder->h, images, g_surrounder->iw, g_surrounder->ih, out_ptr, out_pitch);
    return true;
}

// --- GStreamer & NVMM Specific Implementation ---
#ifdef HAVE_JETSON_NVMM
#include <gst/gst.h>

// 单路处理 (预留，暂时未使用)
bool nvbuffer_cuda_process(GstBuffer *buffer, int width, int height, int brighten_value) {
    return true;
}

// 多路拼接适配器 (核心桥梁)
// 职责：将GStreamer的NVMM Buffer解包为CUDA指针，传给核心算法，再封包回去
bool nvbuffer_cuda_process_multi(GstBuffer **buffers, int width, int height, GstBuffer *out_buffer) {
    if (!buffers || !out_buffer) return false;

    GstMapInfo out_map;
    std::vector<GstMapInfo> in_maps(4);
    
    // 1. 映射输出 Buffer (Write Mode)
    if (!gst_buffer_map(out_buffer, &out_map, GST_MAP_WRITE)) {
        printf("Error: Failed to map output buffer\n");
        return false; 
    }
    NvBufSurface *out_surf = (NvBufSurface *)out_map.data;
    
    // 必须调用 NvBufSurfaceMap 才能在 GPU 上访问 dataPtr
    if (NvBufSurfaceMap(out_surf, 0, -1, NVBUF_MAP_READ_WRITE) != 0) {
        printf("Error: NvBufSurfaceMap failed for output\n");
        gst_buffer_unmap(out_buffer, &out_map);
        return false;
    }
    // 确保之前的 CPU/GPU 操作已完成
    NvBufSurfaceSyncForDevice(out_surf, 0, -1);
    
    // 获取输出显存地址
    uchar4* d_out = (uchar4*)out_surf->surfaceList[0].dataPtr;
    int out_pitch = out_surf->surfaceList[0].pitch;

    // 2. 映射输入 Buffers (Read Mode)
    std::vector<uchar4*> input_ptrs;
    bool map_success = true;
    
    for (int i = 0; i < 4; i++) {
        if (!gst_buffer_map(buffers[i], &in_maps[i], GST_MAP_READ)) {
            printf("Error: Failed to map input buffer %d\n", i);
            map_success = false;
            break;
        }
        NvBufSurface *surf = (NvBufSurface *)in_maps[i].data;
        
        if (NvBufSurfaceMap(surf, 0, -1, NVBUF_MAP_READ) != 0) {
            printf("Error: NvBufSurfaceMap failed for input %d\n", i);
            gst_buffer_unmap(buffers[i], &in_maps[i]); // Clean up this failure
            map_success = false;
            break;
        }
        NvBufSurfaceSyncForDevice(surf, 0, -1);
        
        // 注意：这里假设输入是 RGBA 格式。如果是 NV12，需要核函数支持或前置 nvvidconv 转换
        input_ptrs.push_back((uchar4*)surf->surfaceList[0].dataPtr);
    }

    bool stitching_ok = false;
    if (map_success) {
        // 3. --- 调用核心算法 (通用接口) ---
        // 这一步实现了真正的 "零拷贝"：核心算法直接操作 NVMM 所在的显存
        stitching_ok = stitching_process(d_out, out_pitch, input_ptrs);
    }

    // 4. 清理与解映射 (必须严格执行，否则会导致内存泄漏或死锁)
    
    // 清理输入
    for (int i = 0; i < input_ptrs.size(); i++) { // Only unmap successfully mapped ones
        NvBufSurface *surf = (NvBufSurface *)in_maps[i].data;
        NvBufSurfaceUnMap(surf, 0, -1);
        gst_buffer_unmap(buffers[i], &in_maps[i]);
    }
    
    // 清理输出
    NvBufSurfaceSyncForDevice(out_surf, 0, -1); // 确保内核执行完毕
    NvBufSurfaceUnMap(out_surf, 0, -1);
    gst_buffer_unmap(out_buffer, &out_map);

    return stitching_ok;
}
#else
// x86 存根 (Stubs)
bool nvbuffer_cuda_process(GstBuffer *buffer, int width, int height, int brighten_value) { return false; }
bool nvbuffer_cuda_process_multi(GstBuffer **buffers, int width, int height, GstBuffer *out_buffer) { return false; }
#endif
