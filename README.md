# Jetson RTSP 零拷贝视频处理项目

基于Jetson平台的高性能RTSP流处理项目，实现零拷贝视频处理和CUDA实时加速。

## ✨ 项目特点

- 🚀 **零拷贝架构** - 视频数据始终在GPU显存，无CPU回传
- ⚡ **CUDA实时处理** - 支持自定义CUDA kernel或TensorRT推理
- 📡 **RTSP推流** - 支持多客户端同时连接
- 📦 **开箱即用** - 一键编译运行

## 📊 项目结构

```
.
├── cpp_demo/          # C++实现版本 ⭐推荐
│   ├── src/          # 源代码
│   ├── scripts/      # 构建和运行脚本
│   └── README.md     # 详细文档
├── config/           # 配置文件
```

## 系统依赖

```bash
sudo apt update
sudo apt install -y \
    gstreamer1.0-tools \
    gstreamer1.0-rtsp \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    python3-gi \
    build-essential \
    cmake \
    pkg-config \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    libegl1-mesa-dev \
    libgles2-mesa-dev
```

## C++版本 ⭐ 推荐

### 特性
- ✅ 性能更高，内存占用更低
- ✅ 独立可执行文件，易于部署
- ✅ 统一pipeline架构，无同步问题
- ✅ Identity Hook技术，优雅的CUDA集成
- ✅ 支持多客户端共享

### 快速开始

#### 1. 编译
```bash
cd cpp_demo
bash scripts/build.sh
```

#### 2. 运行
```bash
bash scripts/run.sh
```

#### 3. 拉流测试
```bash
# 在另一台机器上
ffplay rtsp://<JETSON_IP>:8554/live

# 或使用VLC
vlc rtsp://<JETSON_IP>:8554/live
```

### 架构说明
```
V4L2摄像头 → nvvidconv → NVMM → Identity Hook → nvv4l2h264enc → RTSP
                         (显存)      ↓ CUDA处理
```

### 文档
- `cpp_demo/README.md` - 完整使用文档
- `cpp_demo/FINAL_VERSION.md` - 详细技术文档
- `cpp_demo/TROUBLESHOOTING.md` - 问题排查指南
- `cpp_demo/SUMMARY.md` - 项目总结

## 🎯 版本对比

| 特性 | Python版本 | C++版本 |
|------|-----------|---------|
| 性能 | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| 内存占用 | 较高 | 较低 |
| 启动速度 | 较慢 | 快 |
| 易用性 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| 部署 | 需要Python环境 | 独立可执行文件 |
| CUDA集成 | PyCUDA | 原生CUDA |
| 适用场景 | 快速原型 | 生产部署 |
| 推荐度 | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

## 🚀 技术亮点

### 零拷贝架构
- 视频数据从摄像头采集后直接存储在GPU显存（NVMM格式）
- CUDA处理直接在显存上操作，无需CPU回传
- 硬件编码器直接访问显存数据

### CUDA集成方案
- **C++版本**: 使用CUDA External Memory API + Identity Hook

### 无同步问题
- 使用统一的GStreamer pipeline
- Identity element作为CUDA处理hook点
- 避免传统appsrc/appsink的同步复杂性

## 🎓 适用场景

### C++版本适合
- 生产环境部署
- 性能敏感应用
- 嵌入式系统
- 长时间运行的服务

## 🎉 项目状态

- ✅ C++版本 - 功能完整，已优化，生产就绪
- ✅ 文档完善
- ✅ 测试通过

## 📝 License

MIT
