# GEMINI.md

This file provides guidance to Gemini Code (gemini.ai/code) when working with code in this repository.

## Project Overview

This is a Jetson-based **Around View Monitoring (AVM) System** implementing real-time surround view stitching with CUDA acceleration and RTSP streaming. The project supports **two rendering modes**:

- **2D Bird's-Eye View**: Traditional top-down view using Homography matrices (faster, simpler)
- **3D Bowl Rendering**: Immersive 3D surround view using OpenGL ES (advanced, interactive)

### Current Status

**✅ Completed:**
- Python prototype for 2D stitching algorithm (`stitching/`)
- Unified C++ architecture supporting **Sim (x86)** and **Real (Orin)** modes
- CUDA acceleration for 1000x1000 BEV stitching with <1ms kernel latency
- RTSP streaming with integrated AppSrc pipeline
- **Stitching Optimization Phase 1**: Implemented 80px edge feathering, 30° wide blending zones, and closed-loop BGR luminance/color balancing.

**🚧 In Progress (Priority Order):**
1. **True Zero-Copy**: Eliminating GPU-to-CPU host copies by using NVMM-capable encoders and shared memory architectures on Jetson.
2. **Dynamic Extrinsic**: Integrating online calibration algorithms for live pose correction and LUT re-generation.

**📋 Future Planned:**
- 3D surround view with OpenGL ES rendering
- Transparent chassis (Historical frame compensation)

### Architecture

The core architecture captures video from V4L2 cameras, processes frames using CUDA on GPU memory (NVMM), and streams via RTSP with hardware H.264 encoding.

## Build and Run

### C++ Version (Recommended)

**Build:**
```bash
cd cpp_demo
bash scripts/build.sh
```

Or manually:
```bash
cd cpp_demo
mkdir -p build && cd build
cmake ..
make -j$(nproc)
```

**Run:**
```bash
cd cpp_demo
bash scripts/run.sh
# Or directly: ./build/rtsp_demo
```

**Test camera:**
```bash
./build/test_camera
```

### Python Version

**Install dependencies:**
```bash
pip3 install -r requirements.txt
```

**Run:**
```bash
cd python_demo
bash scripts/run_all.sh
```

**View RTSP stream (from another machine):**
```bash
ffplay rtsp://<JETSON_IP>:8554/live
# Or: vlc rtsp://<JETSON_IP>:8554/live
```

## Architecture

### Zero-Copy Pipeline

The key innovation is keeping video data in GPU memory throughout:

```
V4L2 Camera → nvvidconv → NVMM (GPU memory) → Identity Hook → nvv4l2h264enc → RTSP
                                                    ↓
                                              CUDA Processing
```

### C++ Implementation Details

**GStreamer Pipeline:**
- `v4l2src`: Camera capture (YUY2 format)
- `nvvidconv`: Format conversion to NV12 with NVMM allocation
- `identity`: Hook point for CUDA processing (via `signal-handoffs`)
- `nvv4l2h264enc`: Jetson hardware H.264 encoder
- `rtph264pay`: RTP payloader for RTSP streaming

**CUDA Processing Flow (src/nvbuffer_cuda_processor.cu):**
1. Map NVMM buffer to CPU-accessible address via `NvBufSurfaceMap()`
2. Copy Y-plane data to CUDA device memory via `cudaMemcpy(HostToDevice)`
3. Execute CUDA kernel (currently: brightness enhancement)
4. Copy processed data back via `cudaMemcpy(DeviceToHost)`
5. Sync to device with `NvBufSurfaceSyncForDevice()`
6. Unmap buffer with `NvBufSurfaceUnMap()`

This approach uses memory copies but ensures compatibility with the H.264 encoder and maintains stability.

**Main Components:**
- `cpp_demo/src/main.cpp`: GStreamer pipeline setup, RTSP server, signal handling
- `cpp_demo/src/nvbuffer_cuda_processor.cu`: CUDA kernel implementation
- `cpp_demo/src/nvbuffer_cuda_processor.h`: CUDA processor interface
- `cpp_demo/src/test_camera.cpp`: Camera testing utility

### Python Implementation Details

**Architecture:**
- Producer module captures camera frames and processes with CUDA
- RTSP server module handles streaming
- Uses PyCUDA with DMABUF for GPU processing

**Key Files:**
- `python_demo/producer/camera_pipeline.py`: Camera capture pipeline
- `python_demo/producer/cuda_processor.py`: CUDA processing with PyCUDA
- `python_demo/producer/producer.py`: Main producer logic
- `python_demo/rtsp_server/rtsp_server.py`: RTSP streaming server

## Configuration

**Camera settings** are in `config/camera.yaml`:
- `device`: Camera device path (default: `/dev/video1`)
- `width`, `height`: Resolution (default: 1920x1080)
- `framerate`: FPS (default: 30)
- `format`: Pixel format (default: NV12)

**C++ hardcoded settings** in `cpp_demo/src/main.cpp`:
```cpp
static const char *DEVICE = "/dev/video1";  // Camera device
static const int WIDTH = 1920;              // Resolution width
static const int HEIGHT = 1080;             // Resolution height
static const int FPS = 30;                  // Frame rate
```

**CUDA kernel parameters** in `cpp_demo/src/nvbuffer_cuda_processor.cu`:
```cuda
brighten_kernel<<<grid_size, block_size>>>(d_y_plane, y_size, brighten_value);
```
The current kernel adds a brightness value to the Y-plane. Replace this kernel for different processing (e.g., TensorRT inference).

**H.264 encoding parameters** in `cpp_demo/src/main.cpp` pipeline string:
```
nvv4l2h264enc bitrate=8000000 insert-sps-pps=true iframeinterval=30 preset-level=1
```

## Development Notes

### CUDA Integration

The identity element's `handoff` signal callback (`on_identity_handoff()` in main.cpp:76) is where CUDA processing happens. This is the hook point for:
- Custom CUDA kernels
- TensorRT inference
- Other GPU-accelerated processing

### NVMM Memory Handling

NVMM (NVIDIA Multimedia Memory) is GPU memory accessible by hardware encoders. Key APIs:
- `NvBufSurfaceMap()`: Map to CPU-accessible space
- `NvBufSurfaceSyncForDevice()`: Sync CPU changes to GPU
- `NvBufSurfaceUnMap()`: Unmap buffer
- Header: `/usr/src/jetson_multimedia_api/include/nvbufsurface.h`

### CMake Build System

The build requires:
- CUDA toolkit
- GStreamer 1.0 with RTSP server plugin
- Jetson multimedia API (NvBuffer libraries at `/usr/lib/aarch64-linux-gnu/tegra/`)
- EGL and GLESv2 libraries

See `cpp_demo/CMakeLists.txt` for full dependency list.

### Future Development: Dual-Mode AVM System

The project roadmap includes two parallel development paths:

**Path A: 2D Surround View (Priority)**
1. Real vehicle 4-camera calibration (intrinsics + Homography matrices)
2. 2D stitching RTSP streaming integration
3. Performance optimization for real-time processing

**Path B: 3D Surround View (Advanced)**
1. Camera extrinsics calibration (Rotation + Translation)
2. 3D bowl mesh generation
3. OpenGL ES rendering pipeline
4. Interactive view control

See `cpp_demo/docs/TODO.md` for detailed roadmap.

### Platform-Specific Notes

This code is designed for **NVIDIA Jetson** platforms (tested on Orin). It relies on:
- Jetson hardware video encoders (`nvv4l2h264enc`)
- Jetson-specific NvBuffer APIs
- CUDA on Tegra

It will **not** run on x86 systems without significant modifications.

## Testing

**Check camera availability:**
```bash
ls -l /dev/video*
v4l2-ctl --list-devices
v4l2-ctl --device=/dev/video1 --list-formats-ext
```

**Monitor GPU usage:**
```bash
tegrastats        # Jetson-specific
nvidia-smi        # If available
```

**Test RTSP stream latency:**
```bash
ffplay -fflags nobuffer -flags low_delay -framedrop rtsp://<IP>:8554/live
```

## Documentation

Additional documentation in `cpp_demo/docs/`:
- `TODO.md`: Future development roadmap (AVM system)
- `TROUBLESHOOTING.md`: Common issues and solutions
- `PROJECT_STATUS.md`: Current project status
- `DOCS_INDEX.md`: Documentation index
