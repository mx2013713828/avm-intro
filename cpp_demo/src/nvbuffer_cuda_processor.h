#ifndef NVBUFFER_CUDA_PROCESSOR_H
#define NVBUFFER_CUDA_PROCESSOR_H

#include <cuda_runtime.h>
#include <gst/gst.h>

/**
 * 使用NvBuffer API处理NVMM内存中的图像数据
 * @param buffer GStreamer buffer (NVMM格式)
 * @param width 图像宽度
 * @param height 图像高度
 * @param brighten_value 增亮值 (0-255)
 * @return 成功返回true，失败返回false
 */
bool nvbuffer_cuda_process(GstBuffer *buffer, int width, int height, int brighten_value);

/**
 * 初始化CUDA上下文
 */
bool cuda_init();

/**
 * 清理CUDA资源
 */
void cuda_cleanup();

#endif // NVBUFFER_CUDA_PROCESSOR_H

