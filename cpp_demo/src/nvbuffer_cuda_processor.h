#ifndef NVBUFFER_CUDA_PROCESSOR_H
#define NVBUFFER_CUDA_PROCESSOR_H

#include <vector>
#include <cuda_runtime.h>

// Forward declarations
struct _GstBuffer;
typedef struct _GstBuffer GstBuffer;

#ifdef __cplusplus
extern "C" {
#endif

// --- CUDA / Stitching API ---
bool cuda_init();
void cuda_cleanup();

// 加载查找表
bool stitching_init(const char* bin_file, int out_w, int out_h, int in_w, int in_h);

// 核心拼接函数 (输入输出都在GPU显存中)
// gains: 12个float, 格式为 [B0, G0, R0, B1, G1, R1, B2, G2, R2, B3, G3, R3]
bool stitching_process(uchar4* out_ptr, int out_pitch, const std::vector<uchar4*>& in_ptrs, const float* gains = nullptr);

// --- Memory Management for Zero-Copy ---
void* nvbuffer_allocate_output(int w, int h, int* out_pitch);
void  nvbuffer_free_output(void* surf);
uchar4* nvbuffer_map_to_cuda(void* surf, int* out_pitch);
void  nvbuffer_unmap_from_cuda(void* surf);
GstBuffer* nvbuffer_wrap_as_gstbuffer(void* surf, int w, int h);

// --- GStreamer Adapters ---
bool nvbuffer_cuda_process(GstBuffer *buffer, int width, int height, int brighten_value);
bool nvbuffer_cuda_process_multi(GstBuffer **buffers, int width, int height, GstBuffer *out_buffer);

#ifdef __cplusplus
}
#endif

#endif // NVBUFFER_CUDA_PROCESSOR_H
