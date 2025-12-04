import ctypes
import numpy as np

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst

# CUDA / EGL imports
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

import pycuda.gl
from pycuda.gl import RegisteredBuffer

import ctypes
import sys

# EGL imports
from pycuda.tools import context_dependent_memoize
import pycuda._driver as _drv


###############################################
# CUDA 内核：增加亮度（示例）
###############################################

kernel_code = r"""
extern "C" __global__
void brighten(unsigned char* img, int size, int value)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        int v = img[idx] + value;
        img[idx] = v > 255 ? 255 : v;
    }
}
"""

mod = SourceModule(kernel_code)
kernel_brighten = mod.get_function("brighten")


###############################################
# DMABUF → CUDA pointer → brightness kernel
###############################################
def cuda_process_dmabuf(dmabuf_fd, width, height):
    """
    完整 CUDA 零拷贝处理：
    1. 使用 dmabuf fd 创建 EGLImage
    2. 将 EGLImage 映射为 CUDA pointer
    3. 执行 CUDA kernel（示例：增加亮度 +40）
    """

    import pycuda.driver as cuda
    import pycuda.gl as cudagl
    from pycuda.gl import RegisteredBuffer

    # ---- Step 1: DMABUF Fd → EGLImage ----
    try:
        egl_image = cuda.EGLImage.from_dmabuf(
            dmabuf_fd,
            width,
            height,
            cuda.EGLImageColorFormat.NV12
        )
    except Exception as e:
        print("Failed create EGLImage:", e)
        return False

    # ---- Step 2: EGLImage → CUDA pointer ----
    try:
        cuda_resource = cuda.graphics_egl_register_image(egl_image)
        cuda_resource.map()
        ptr, size = cuda_resource.get_mapped_pointer()
    except Exception as e:
        print("CUDA mapping failed:", e)
        return False

    # ---- Step 3: Launch CUDA kernel ----
    # NV12: Y plane first = width*height bytes
    Y_size = width * height

    block = (256, 1, 1)
    grid = ((Y_size + 255) // 256, 1, 1)

    try:
        # 只增强 Y plane（亮度）
        kernel_brighten(
            ptr,
            np.int32(Y_size),
            np.int32(40),      # 增亮量，可调
            block=block,
            grid=grid
        )
    except Exception as e:
        print("CUDA kernel failed:", e)

    # ---- Step 4: Unmap ----
    cuda_resource.unmap()
    cuda_resource.unregister()

    return True

