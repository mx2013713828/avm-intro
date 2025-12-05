# <h1 align="center">Jetson RTSP 零拷贝视频处理项目</p>

<p align="center">
  <a href="https://bun.com"><img src="https://pic.imgdb.cn/item/65dc5dfc9f345e8d03446103.png" align="center" width="220" height="82"></a>
</p>

#### <p align = "center">![Static Badge](https://img.shields.io/badge/mayufeng-blue?style=flat&label=Author)![Static Badge](https://img.shields.io/badge/2025/12/01-blue?style=flat&label=CreateTime)![Static Badge](https://img.shields.io/badge/97357473@qq\.com\-blue?style=flat&label=Email)</p>

基于Jetson平台的高性能RTSP流处理项目，实现零拷贝视频处理和CUDA实时加速。

## ✨ 项目特点

- 🚀 **零拷贝架构** - 视频数据始终在GPU显存，无CPU回传（目前为伪零拷贝）
- ⚡ **CUDA实时处理** - 支持自定义CUDA kernel或TensorRT推理
- 📡 **RTSP推流** - 支持多客户端同时连接
- 📦 **开箱即用** - 一键编译运行

## 📁 项目结构

```
.
├── cpp_demo/           # [核心] C++ 高性能实现版本 (Jetson Orin)
│   ├── src/            # 源代码 (GStreamer + CUDA)
│   ├── scripts/        # 编译运行脚本
│   └── README.md       # C++版本详细文档
├── stitching/          # [算法] Python 环视拼接算法原型
│   ├── generate_data.py # 离线参数生成工具
│   └── cuda.cu         # CUDA核函数原型
├── config/             # 配置文件
└── README.md           # 项目总览
```

## 🚀 快速开始

本项目包含两个主要部分，请根据需求选择：

### 1. C++ 高性能版本 (推荐)
适用于 **Jetson Orin** 等嵌入式平台的生产环境部署。
- **特点**: C++ / GStreamer / CUDA / RTSP
- **文档**: 请阅读 [cpp_demo/README.md](cpp_demo/README.md)

### 2. Python 算法原型
适用于算法研究、验证和离线数据生成。
- **特点**: Python / OpenCV / PyTorch
- **位置**: `stitching/` 目录

## 📊 性能指标

| 代码实现 | 耗时 (ms/帧) | CPU 使用率 |延迟(ms)|
| :------- | :---------- | :--------- |:--------- |
| C++ & CUDA | 15       | 30%      | 200~250 |

## ⚙️ 开发环境

- **硬件**: NVIDIA Jetson Orin / AGX Xavier
- **系统**: Ubuntu 20.04 (JetPack 5.1)
- **依赖**: CUDA, TensorRT, GStreamer, OpenCV

## 📝 License

MIT
