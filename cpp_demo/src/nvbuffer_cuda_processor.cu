#include "nvbuffer_cuda_processor.h"
#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <chrono>
#include <gst/gst.h>
#include <nvbufsurface.h>
#include <nvbufsurftransform.h>

// CUDA内核：增加亮度（极端效果用于对比）
__global__ void brighten_kernel(unsigned char* img, int size, int value)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        int v = img[idx] + value;
        img[idx] = v > 255 ? 255 : (v < 0 ? 0 : v);
    }
}

static CUcontext cuda_context = nullptr;
static CUdevice cuda_device = 0;

bool cuda_init() {
    CUresult err = cuInit(0);
    if (err != CUDA_SUCCESS) {
        const char* error_str;
        cuGetErrorString(err, &error_str);
        printf("Failed to initialize CUDA: %s\n", error_str);
        return false;
    }

    err = cuDeviceGet(&cuda_device, 0);
    if (err != CUDA_SUCCESS) {
        const char* error_str;
        cuGetErrorString(err, &error_str);
        printf("Failed to get CUDA device: %s\n", error_str);
        return false;
    }

    err = cuCtxCreate(&cuda_context, 0, cuda_device);
    if (err != CUDA_SUCCESS) {
        const char* error_str;
        cuGetErrorString(err, &error_str);
        printf("Failed to create CUDA context: %s\n", error_str);
        return false;
    }

    printf("CUDA initialized successfully\n");
    return true;
}

void cuda_cleanup() {
    if (cuda_context) {
        cuCtxDestroy(cuda_context);
        cuda_context = nullptr;
    }
}

bool nvbuffer_cuda_process(GstBuffer *buffer, int width, int height, int brighten_value) {
    static int process_count = 0;
    static auto last_print_time = std::chrono::steady_clock::now();
    process_count++;
    
    // 从GstBuffer获取NvBufSurface
    GstMapInfo map_info;
    if (!gst_buffer_map(buffer, &map_info, (GstMapFlags)(GST_MAP_READ | GST_MAP_WRITE))) {
        printf("Failed to map buffer\n");
        return false;
    }
    
    NvBufSurface *surf = (NvBufSurface *)map_info.data;
    
    if (!surf || surf->numFilled == 0 || surf->surfaceList == nullptr) {
        gst_buffer_unmap(buffer, &map_info);
        return false;
    }
    
    NvBufSurfaceParams *params = &surf->surfaceList[0];
    
    if (process_count == 1) {
        printf("NvBufSurface: memType=%d, colorFormat=%d, size=%dx%d\n", 
               surf->memType, params->colorFormat, params->width, params->height);
    }
    
    // 方法：直接在原buffer上操作，但要小心同步
    // 映射surface到CUDA可访问内存
    if (NvBufSurfaceMap(surf, 0, -1, NVBUF_MAP_READ_WRITE) != 0) {
        if (process_count == 1) {
            printf("Failed to map NvBufSurface\n");
        }
        gst_buffer_unmap(buffer, &map_info);
        return false;
    }
    
    // 同步到设备
    NvBufSurfaceSyncForDevice(surf, 0, -1);
    
    // 获取第一个平面（Y平面）的地址
    void *y_plane_addr = nullptr;
    
    // 对于NVMM内存，使用mappedAddr来访问
    if (params->mappedAddr.addr[0]) {
        y_plane_addr = params->mappedAddr.addr[0];
    } else {
        if (process_count == 1) {
            printf("No valid mapped address\n");
        }
        NvBufSurfaceUnMap(surf, 0, -1);
        gst_buffer_unmap(buffer, &map_info);
        return false;
    }
    
    if (process_count == 1) {
        printf("✓ Y-plane mapped at: %p\n", y_plane_addr);
    }
    
    // 将CPU地址的数据拷贝到CUDA设备
    int y_size = width * height;
    
    // 分配CUDA设备内存
    unsigned char *d_y_plane = nullptr;
    cudaError_t cuda_err = cudaMalloc(&d_y_plane, y_size);
    if (cuda_err != cudaSuccess) {
        printf("cudaMalloc failed: %s\n", cudaGetErrorString(cuda_err));
        NvBufSurfaceUnMap(surf, 0, -1);
        gst_buffer_unmap(buffer, &map_info);
        return false;
    }
    
    // 从mapped地址拷贝到CUDA内存
    cuda_err = cudaMemcpy(d_y_plane, y_plane_addr, y_size, cudaMemcpyHostToDevice);
    if (cuda_err != cudaSuccess) {
        printf("cudaMemcpy H2D failed: %s\n", cudaGetErrorString(cuda_err));
        cudaFree(d_y_plane);
        NvBufSurfaceUnMap(surf, 0, -1);
        gst_buffer_unmap(buffer, &map_info);
        return false;
    }
    
    // 配置kernel参数
    int block_size = 256;
    int grid_size = (y_size + block_size - 1) / block_size;
    
    // 启动kernel处理
    brighten_kernel<<<grid_size, block_size>>>(d_y_plane, y_size, brighten_value);
    
    // 同步
    cuda_err = cudaDeviceSynchronize();
    if (cuda_err != cudaSuccess) {
        printf("CUDA kernel failed: %s\n", cudaGetErrorString(cuda_err));
        cudaFree(d_y_plane);
        NvBufSurfaceUnMap(surf, 0, -1);
        gst_buffer_unmap(buffer, &map_info);
        return false;
    }
    
    // 拷贝处理后的数据回mapped地址
    cuda_err = cudaMemcpy(y_plane_addr, d_y_plane, y_size, cudaMemcpyDeviceToHost);
    if (cuda_err != cudaSuccess) {
        printf("cudaMemcpy D2H failed: %s\n", cudaGetErrorString(cuda_err));
        cudaFree(d_y_plane);
        NvBufSurfaceUnMap(surf, 0, -1);
        gst_buffer_unmap(buffer, &map_info);
        return false;
    }
    
    // 释放CUDA内存
    cudaFree(d_y_plane);
    
    // 同步回设备
    NvBufSurfaceSyncForDevice(surf, 0, -1);
    
    // 取消映射
    NvBufSurfaceUnMap(surf, 0, -1);
    
    // Unmap buffer
    gst_buffer_unmap(buffer, &map_info);
    
    // 统计
    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - last_print_time).count();
    if (elapsed >= 1) {
        printf("✓ [CUDA] Processed %d frames (Brightness +%d, Resolution: %dx%d)\n", 
               process_count, brighten_value, width, height);
        last_print_time = now;
        process_count = 0;
    }
    
    return true;
}
