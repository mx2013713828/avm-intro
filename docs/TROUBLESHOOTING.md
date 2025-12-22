# 🔧 故障排查指南

本文档提供常见问题的解决方案和调试技巧。

---

## 📋 目录

1. [摄像头问题](#摄像头问题)
2. [编译问题](#编译问题)
3. [CUDA问题](#cuda问题)
4. [RTSP推流问题](#rtsp推流问题)
5. [性能问题](#性能问题)
6. [调试工具](#调试工具)

---

## 摄像头问题

### ❌ 问题: 找不到摄像头设备

**错误信息**:
```
ERROR from v4l2src0: Cannot identify device '/dev/video0'
```

**解决方案**:

1. **检查设备是否存在**:
```bash
ls -l /dev/video*
# 应该看到类似: /dev/video0, /dev/video1 等
```

2. **检查设备信息**:
```bash
v4l2-ctl --list-devices
# 查看所有摄像头设备及其驱动信息
```

3. **检查设备权限**:
```bash
sudo chmod 666 /dev/video0
# 或者将用户添加到video组
sudo usermod -a -G video $USER
# 注销后重新登录生效
```

4. **测试摄像头**:
```bash
# 使用test_camera工具测试
./build/test_camera

# 或使用GStreamer直接测试
gst-launch-1.0 v4l2src device=/dev/video0 ! videoconvert ! autovideosink
```

---

### ❌ 问题: 摄像头分辨率不支持

**错误信息**:
```
ERROR: Caps negotiation failed
```

**解决方案**:

1. **查询摄像头支持的格式**:
```bash
v4l2-ctl --device=/dev/video0 --list-formats-ext
```

2. **修改main.cpp中的配置**:
```cpp
// 根据摄像头支持的格式修改
static const int WIDTH = 1280;   // 改为支持的宽度
static const int HEIGHT = 720;   // 改为支持的高度
static const int FPS = 30;
```

3. **常见分辨率**:
- 1920×1080 (Full HD)
- 1280×720 (HD)
- 640×480 (VGA)

---

### ❌ 问题: Opening in BLOCKING MODE

**现象**:
```
Opening in BLOCKING MODE
```

这是正常信息，不是错误。表示V4L2以阻塞模式打开摄像头。

---

## 编译问题

### ❌ 问题: cuda_runtime.h not found

**错误信息**:
```
fatal error: cuda_runtime.h: No such file or directory
```

**解决方案**:

1. **检查CUDA安装**:
```bash
nvcc --version
ls -l /usr/local/cuda
```

2. **确保CMakeLists.txt包含正确路径**:
```cmake
include_directories(
    /usr/local/cuda/include
)
```

3. **设置环境变量**:
```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

---

### ❌ 问题: 找不到GStreamer库

**错误信息**:
```
Could not find a package configuration file provided by "GStreamer"
```

**解决方案**:

```bash
# 安装GStreamer开发包
sudo apt install -y \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    gstreamer1.0-rtsp
```

---

### ❌ 问题: nvbufsurface.h not found

**错误信息**:
```
fatal error: nvbufsurface.h: No such file or directory
```

**解决方案**:

1. **检查头文件位置**:
```bash
find /usr -name "nvbufsurface.h"
# 通常在: /usr/src/jetson_multimedia_api/include/
```

2. **确保CMakeLists.txt包含路径**:
```cmake
include_directories(
    /usr/src/jetson_multimedia_api/include
)
```

---

## CUDA问题

### ❌ 问题: CUDA kernel execution failed: an illegal memory access

**错误信息**:
```
CUDA kernel execution failed: an illegal memory access was encountered
```

**可能原因**:
1. 指针访问越界
2. 使用了未映射的内存地址
3. 内存未正确同步

**解决方案**:

当前版本已修复此问题，使用安全的内存拷贝方法：

```cpp
// 正确的做法：
// 1. 映射内存
NvBufSurfaceMap(surf, 0, -1, NVBUF_MAP_READ_WRITE);

// 2. 拷贝到CUDA设备
cudaMemcpy(d_ptr, mapped_addr, size, cudaMemcpyHostToDevice);

// 3. 执行kernel
my_kernel<<<grid, block>>>(d_ptr, ...);

// 4. 拷贝回去
cudaMemcpy(mapped_addr, d_ptr, size, cudaMemcpyDeviceToHost);

// 5. 同步并取消映射
NvBufSurfaceSyncForDevice(surf, 0, -1);
NvBufSurfaceUnMap(surf, 0, -1);
```

---

### ❌ 问题: NvBufSurfaceSyncForCpu failed

**错误信息**:
```
NvMapMemCacheMaint Bad parameter
nvbusurface: NvBufSurfaceSyncForCpu: Error(4) in sync
```

**原因**: NVMM内存已经在GPU上，不需要sync到CPU

**解决方案**: 已在当前版本中修复，不再调用不必要的sync

---

### ❌ 问题: CUDA处理率为0

**现象**:
```
✓ Processed 30 frames | CUDA: NOT ACTIVE (all skipped)
```

**可能原因**:
1. CUDA初始化失败
2. 内存映射失败
3. Buffer格式不正确

**调试步骤**:

1. **检查第一帧的调试输出**:
```
NvBufSurface info: memType=4, numFilled=1, colorFormat=6
✓ Y-plane mapped at: 0x...
```

2. **如果看不到映射信息，检查代码逻辑**

3. **增加调试输出**:
```cpp
printf("Debug: buffer=%p, surf=%p\n", buffer, surf);
```

---

## RTSP推流问题

### ❌ 问题: Service Unavailable (503)

**错误信息**:
```
RTSP/1.0 503 Service Unavailable
```

**可能原因**:
1. Pipeline未正常启动
2. 摄像头未打开
3. 编码器失败

**解决方案**:

1. **检查服务器日志**:
```bash
# 运行rtsp_demo时查看输出
./build/rtsp_demo
# 应该看到: "Waiting for RTSP clients..."
```

2. **测试摄像头**:
```bash
./build/test_camera
```

3. **检查端口占用**:
```bash
netstat -tuln | grep 8554
# 如果被占用，修改端口或kill进程
```

---

### ❌ 问题: 拉流画面卡顿或花屏

**可能原因**:
1. 网络带宽不足
2. 码率过高
3. 播放器缓冲设置不当

**解决方案**:

1. **降低码率**:
编辑 `src/main.cpp`:
```cpp
"nvv4l2h264enc bitrate=4000000"  // 从8M降到4M
```

2. **使用低延迟播放器参数**:

**VLC**:
```
设置 → 输入/编解码器 → 网络缓存 → 设为300ms
```

**GStreamer**:
```bash
gst-launch-1.0 rtspsrc location=rtsp://IP:8554/live latency=0 ! \
    decodebin ! videoconvert ! autovideosink sync=false
```

**FFplay**:
```bash
ffplay -fflags nobuffer -flags low_delay -framedrop \
    -probesize 32 -analyzeduration 0 \
    rtsp://IP:8554/live
```

3. **检查网络延迟**:
```bash
ping <jetson_ip>
# 延迟应该 < 10ms
```

---

### ❌ 问题: 客户端连接后立即断开

**现象**:
```
New RTSP client connected
RTSP client connected - Media configured
[客户端立即断开]
```

**可能原因**:
1. Pipeline启动失败
2. 编码器配置不兼容

**解决方案**:

1. **检查编码器输出**:
```bash
# 查看是否有编码错误
# 运行时注意这些消息:
# H264: Profile = 66, Level = 0
# NVMEDIA: Need to set EMC bandwidth : 846000
```

2. **简化pipeline测试**:
```bash
# 直接测试编码和推流（不使用CUDA）
gst-launch-1.0 v4l2src device=/dev/video0 ! \
    nvvidconv ! 'video/x-raw(memory:NVMM),format=NV12' ! \
    nvv4l2h264enc ! h264parse ! rtph264pay ! \
    udpsink host=127.0.0.1 port=5000
```

---

## 性能问题

### ❌ 问题: 帧率低于30fps

**现象**:
```
✓ [CUDA] Processed 15 frames (should be ~30)
```

**可能原因**:
1. CUDA处理过慢
2. CPU/GPU负载过高
3. 内存带宽瓶颈

**解决方案**:

1. **监控系统资源**:
```bash
# Jetson监控工具
tegrastats

# 或使用jtop
sudo pip3 install jetson-stats
sudo jtop
```

2. **降低分辨率**:
```cpp
static const int WIDTH = 1280;   // 从1920降低
static const int HEIGHT = 720;   // 从1080降低
```

3. **优化CUDA kernel**:
- 增加block size
- 使用shared memory
- 减少内存传输

---

### ❌ 问题: GPU占用率100%

**现象**: `tegrastats` 显示GPU使用率持续100%

**分析**:
- 如果帧率正常（30fps），这是正常的，说明GPU充分利用
- 如果帧率低，说明需要优化

**解决方案**:

1. **降低处理负载**:
- 降低分辨率
- 简化CUDA算法

2. **使用异步处理**:
```cpp
// 使用CUDA Streams
cudaStream_t stream;
cudaStreamCreate(&stream);
my_kernel<<<grid, block, 0, stream>>>(d_ptr);
```

---

### ❌ 问题: 功耗过高或过热

**现象**: 设备发热严重，或触发温度保护

**解决方案**:

1. **检查温度**:
```bash
tegrastats | grep temp
```

2. **降低功耗模式**:
```bash
# 查看当前功耗模式
sudo /usr/sbin/nvpmodel -q

# 设置为更节能的模式
sudo /usr/sbin/nvpmodel -m <mode_id>
```

3. **添加散热**:
- 安装散热风扇
- 使用散热片

---

## 调试工具

### GStreamer调试

**启用GST_DEBUG**:
```bash
# 设置调试级别 (0-9, 9最详细)
export GST_DEBUG=3
./build/rtsp_demo

# 只显示特定组件
export GST_DEBUG=v4l2:5,nvvidconv:5
./build/rtsp_demo

# 保存到文件
export GST_DEBUG=3
export GST_DEBUG_FILE=/tmp/gst_debug.log
./build/rtsp_demo
```

---

### CUDA调试

**检查CUDA错误**:
```cpp
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA Error: %s at %s:%d\n", \
                   cudaGetErrorString(err), __FILE__, __LINE__); \
        } \
    } while(0)

// 使用
CUDA_CHECK(cudaMalloc(&d_ptr, size));
```

**cuda-memcheck**:
```bash
# 检查内存错误
cuda-memcheck ./build/rtsp_demo
```

---

### 性能分析

**使用nvprof** (CUDA 11.4可能不支持，使用Nsight):
```bash
# 简单profiling
nvprof ./build/rtsp_demo

# 或使用Nsight Systems
nsys profile -o report ./build/rtsp_demo
```

**手动计时**:
```cpp
auto start = std::chrono::high_resolution_clock::now();
// ... 你的代码 ...
auto end = std::chrono::high_resolution_clock::now();
auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
printf("Time: %ld ms\n", duration.count());
```

---

### 网络调试

**抓包分析**:
```bash
# 抓取RTSP流
sudo tcpdump -i any -w rtsp.pcap port 8554

# 使用Wireshark分析
wireshark rtsp.pcap
```

**测试RTSP连接**:
```bash
# 使用ffprobe查看流信息
ffprobe -rtsp_transport tcp rtsp://192.168.1.100:8554/live
```

---

## 日志分析

### 正常运行的日志

```
Initializing CUDA...
CUDA initialized successfully
RTSP Server started
Waiting for RTSP clients...

[客户端连接后]
New RTSP client connected
RTSP client connected - Media configured
CUDA processing hook installed
Opening in BLOCKING MODE
NvMMLiteOpen : Block : BlockType = 4
===== NVMEDIA: NVENC =====
NvBufSurface info: memType=4, numFilled=1, colorFormat=6
✓ Y-plane mapped at: 0x...
H264: Profile = 66, Level = 0
NVMEDIA: Need to set EMC bandwidth : 846000

[运行中]
✓ Processed 30 frames | CUDA: 30 success, 0 skipped
✓ [CUDA] Processed 20 frames (Brightness +80, Resolution: 1920x1080)
```

### 异常日志标识

❌ **错误标识**:
- `ERROR from`
- `failed`
- `Error(`
- `Cannot`

⚠️ **警告标识**:
- `Warning:`
- `Failed to` (某些可以忽略)

---

## 常用命令速查

```bash
# 摄像头
ls -l /dev/video*
v4l2-ctl --list-devices
v4l2-ctl --device=/dev/video0 --list-formats-ext

# 编译
./scripts/build.sh
cd build && make -j$(nproc)

# 运行
./build/rtsp_demo
./build/test_camera

# 系统监控
tegrastats
sudo jtop
nvidia-smi  # (某些Jetson不支持)

# 网络
netstat -tuln | grep 8554
ifconfig
ping <target_ip>

# CUDA
nvcc --version
cuda-memcheck ./build/rtsp_demo

# GStreamer
export GST_DEBUG=3
gst-inspect-1.0 nvvidconv
gst-launch-1.0 --gst-debug=3 ...
```

---

## 寻求帮助

如果以上方法都无法解决问题：

1. **收集信息**:
   - 错误日志（完整的终端输出）
   - 系统信息：`uname -a`, `nvcc --version`
   - 设备信息：`v4l2-ctl --list-devices`

2. **提交Issue**:
   - 清楚描述问题
   - 提供复现步骤
   - 附上日志和配置

3. **查阅文档**:
   - [GStreamer文档](https://gstreamer.freedesktop.org/documentation/)
   - [Jetson Linux文档](https://docs.nvidia.com/jetson/)
   - [CUDA文档](https://docs.nvidia.com/cuda/)

---

**最后更新**: 2025-11-27
