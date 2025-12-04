# 🎯 四路环视拼接系统 - 开发计划

## 📋 项目目标

实现一个基于CUDA的实时四路摄像头环视拼接系统，支持RTSP推流输出。

## 🏗️ 系统架构设计

```
┌─────────────────────────────────────────────────────────────┐
│                      四路摄像头输入                          │
│  Camera 0      Camera 1      Camera 2      Camera 3        │
│  (前视)        (右视)        (后视)        (左视)           │
│    ↓             ↓             ↓             ↓              │
├─────────────────────────────────────────────────────────────┤
│                    V4L2 + nvvidconv                         │
│              NVMM Buffer (四路独立buffer)                   │
│    ↓             ↓             ↓             ↓              │
├─────────────────────────────────────────────────────────────┤
│                    CUDA 环视拼接处理                         │
│  • 畸变校正 (Undistortion)                                  │
│  • 透视变换 (Perspective Transform)                         │
│  • 图像融合 (Blending)                                      │
│  • 鸟瞰图生成 (Bird's Eye View)                             │
│    ↓                                                        │
├─────────────────────────────────────────────────────────────┤
│              拼接后的单路输出 (NVMM)                         │
│    ↓                                                        │
├─────────────────────────────────────────────────────────────┤
│          H.264编码 (nvv4l2h264enc)                          │
│    ↓                                                        │
├─────────────────────────────────────────────────────────────┤
│              RTSP Server 推流                               │
└─────────────────────────────────────────────────────────────┘
```

## 📅 开发阶段

### ✅ Phase 0: 单路处理基础 (已完成)

**目标**: 实现单路摄像头的CUDA处理和RTSP推流

**完成项**:
- [x] 单路V4L2摄像头采集
- [x] NVMM零拷贝内存管理
- [x] NvBufSurface API集成
- [x] CUDA kernel基础处理 (亮度调整)
- [x] H.264硬件编码
- [x] RTSP Server推流
- [x] 性能验证: 30fps @ 1920x1080

**经验总结**:
- NVMM内存需要通过 `NvBufSurfaceMap` 映射
- 使用 `cudaMemcpy` 在CPU和GPU间传输数据
- `identity` element作为CUDA处理hook点
- 需要正确同步以避免编码器冲突

---

### 🔄 Phase 1: 四路同步采集

**目标**: 实现四路摄像头的同步采集和独立pipeline管理

**任务列表**:

#### 1.1 硬件配置
- [ ] 确认四个摄像头设备路径 (`/dev/video0-3`)
- [ ] 验证每个摄像头的分辨率和帧率
- [ ] 测试USB/CSI带宽是否足够支持四路同时采集
- [ ] 确定每路摄像头的物理位置（前/后/左/右）

#### 1.2 多路Pipeline架构
- [ ] 设计四路独立GStreamer pipeline
- [ ] 实现pipeline同步机制（时间戳对齐）
- [ ] 创建四路buffer管理器
- [ ] 实现帧同步检测（确保四路帧时间戳一致）

#### 1.3 代码结构
```
src/
├── main.cpp                          # 主程序
├── camera_manager.h/cpp              # 摄像头管理器
├── multi_pipeline.h/cpp              # 多路pipeline管理
├── frame_sync.h/cpp                  # 帧同步器
└── cuda/
    └── avm_processor.h/cu            # 环视CUDA处理器
```

**预期输出**:
- 四路视频成功同步采集
- 帧率稳定在30fps
- 时间戳误差 < 33ms (1帧)

---

### 🎨 Phase 2: 相机标定与校正

**目标**: 实现相机标定和畸变校正

#### 2.1 相机标定
- [ ] 准备棋盘格标定板
- [ ] 采集每个摄像头的标定图像（20-30张）
- [ ] 使用OpenCV标定工具获取相机内参
- [ ] 获取畸变系数（k1, k2, p1, p2, k3）
- [ ] 保存标定参数到配置文件

**标定参数格式**:
```yaml
camera_0:  # 前视
  intrinsic:
    fx: 1000.0
    fy: 1000.0
    cx: 960.0
    cy: 540.0
  distortion:
    k1: -0.2
    k2: 0.1
    p1: 0.0
    p2: 0.0
    k3: 0.0
  resolution: [1920, 1080]
  position: "front"
```

#### 2.2 CUDA畸变校正
- [ ] 实现CUDA畸变校正kernel
- [ ] 使用look-up table (LUT) 加速
- [ ] 预计算映射表并上传到GPU
- [ ] 优化内存访问模式（texture memory）
- [ ] 性能测试: 目标 < 5ms/帧

**关键技术**:
```cuda
// 畸变校正kernel示例
__global__ void undistort_kernel(
    unsigned char* dst,
    const unsigned char* src,
    const float* map_x,
    const float* map_y,
    int width, int height
);
```

---

### 🗺️ Phase 3: 透视变换与鸟瞰图

**目标**: 将四路图像变换到鸟瞰视角

#### 3.1 地面标定
- [ ] 在车辆周围绘制标准网格（如1m × 1m）
- [ ] 采集四路摄像头的地面网格图像
- [ ] 手动标注关键点（至少4个点/相机）
- [ ] 计算透视变换矩阵（Homography）

#### 3.2 CUDA透视变换
- [ ] 实现CUDA透视变换kernel
- [ ] 支持双线性插值
- [ ] 优化边界处理
- [ ] 性能目标: < 3ms/帧/路

**透视变换公式**:
```
[x']   [h00 h01 h02]   [x]
[y'] = [h10 h11 h12] × [y]
[w']   [h20 h21 h22]   [1]

x_bird = x' / w'
y_bird = y' / w'
```

#### 3.3 鸟瞰图布局
- [ ] 设计输出图像尺寸（如2048×2048）
- [ ] 定义四路图像在输出中的位置
- [ ] 计算重叠区域（用于融合）
- [ ] 可视化调试工具

---

### 🌈 Phase 4: 多视图融合

**目标**: 平滑融合四路图像的重叠区域

#### 4.1 融合策略
- [ ] 实现简单加权融合（alpha blending）
- [ ] 实现渐变融合（gradient blending）
- [ ] 实现泊松融合（可选，高级）
- [ ] 处理亮度差异（直方图匹配）

#### 4.2 CUDA融合kernel
```cuda
// 多视图融合kernel
__global__ void blend_kernel(
    unsigned char* output,
    const unsigned char* cam0,
    const unsigned char* cam1,
    const unsigned char* cam2,
    const unsigned char* cam3,
    const float* weight_maps,
    int width, int height
);
```

#### 4.3 权重图生成
- [ ] 基于距离的权重计算
- [ ] 考虑图像质量（清晰度、亮度）
- [ ] 预计算权重图并上传GPU
- [ ] 动态权重调整（可选）

---

### 🔧 Phase 5: 性能优化

**目标**: 达到实时处理要求（30fps @ 1920x1080输入 → 2048x2048输出）

#### 5.1 内存优化
- [ ] 使用Pinned Memory加速CPU-GPU传输
- [ ] 使用CUDA Texture Memory优化随机访问
- [ ] 实现double buffering避免等待
- [ ] 减少不必要的内存拷贝

#### 5.2 计算优化
- [ ] 使用CUDA Streams并发处理
- [ ] 合并kernel减少启动开销
- [ ] 优化线程块大小（block/grid configuration）
- [ ] 使用Shared Memory缓存热点数据

#### 5.3 Pipeline优化
- [ ] 四路采集并行化
- [ ] CUDA处理与编码流水线化
- [ ] 异步处理避免阻塞

**性能指标**:
| 阶段 | 目标延迟 | 实际延迟 |
|------|---------|---------|
| 四路采集 | < 10ms | TBD |
| 畸变校正 | < 20ms (4×5ms) | TBD |
| 透视变换 | < 12ms (4×3ms) | TBD |
| 图像融合 | < 10ms | TBD |
| H.264编码 | < 15ms | TBD |
| **总延迟** | **< 67ms** | **TBD** |

---

### 🚀 Phase 6: 系统集成与测试

**目标**: 集成完整系统并进行全面测试

#### 6.1 功能集成
- [ ] 集成所有CUDA处理模块
- [ ] 实现配置文件加载
- [ ] 添加命令行参数支持
- [ ] 实现运行时参数调整

#### 6.2 错误处理
- [ ] 摄像头断线检测与恢复
- [ ] CUDA错误检测与降级
- [ ] 编码失败处理
- [ ] 日志系统完善

#### 6.3 测试用例
- [ ] 单元测试（每个模块）
- [ ] 集成测试（完整pipeline）
- [ ] 性能测试（帧率、延迟、CPU/GPU占用）
- [ ] 压力测试（长时间运行）
- [ ] 边界测试（摄像头遮挡、光照变化）

#### 6.4 文档完善
- [ ] API文档
- [ ] 用户手册
- [ ] 标定指南
- [ ] 故障排查指南

---

## 🛠️ 技术选型

### 核心库
- **GStreamer 1.x**: 视频pipeline管理
- **CUDA 11.4**: GPU加速计算
- **NvBuffer API**: NVMM内存管理
- **OpenCV** (可选): 标定工具、离线处理
- **yaml-cpp**: 配置文件解析

### 硬件要求
- **平台**: NVIDIA Jetson Orin
- **摄像头**: 4×USB/CSI摄像头
- **内存**: 建议 ≥ 8GB
- **存储**: ≥ 16GB (用于标定数据)

---

## 📊 性能目标

| 指标 | 目标值 |
|------|--------|
| 输入分辨率 | 4 × 1920×1080 @ 30fps |
| 输出分辨率 | 2048×2048 @ 30fps |
| 端到端延迟 | < 100ms |
| GPU占用率 | < 80% |
| 功耗 | < 25W |

---

## 🔍 风险与挑战

### 技术风险
1. **多路同步**: 四路摄像头时间戳对齐困难
   - 缓解: 使用硬件时间戳，容忍小误差
   
2. **计算性能**: CUDA处理可能无法达到30fps
   - 缓解: 降低分辨率或帧率，优化算法
   
3. **内存带宽**: 四路视频数据量大
   - 缓解: 使用NVMM零拷贝，减少传输

### 硬件风险
1. **USB带宽限制**: USB 3.0可能无法支持4×1080p
   - 缓解: 使用CSI接口，或降低部分摄像头分辨率
   
2. **摄像头质量**: 不同摄像头色彩不一致
   - 缓解: 添加颜色校准步骤

---

## 📚 参考资料

### 环视拼接算法
- [ ] [Around View Monitor (AVM) System](https://en.wikipedia.org/wiki/Around_view_monitor)
- [ ] [OpenCV Camera Calibration](https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html)
- [ ] [NVIDIA VPI - Perspective Warp](https://docs.nvidia.com/vpi/algorithms.html)

### CUDA优化
- [ ] [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [ ] [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)

### Jetson多媒体
- [ ] [Jetson Linux Multimedia API](https://docs.nvidia.com/jetson/l4t-multimedia/index.html)
- [ ] [GStreamer on Jetson](https://developer.ridgerun.com/wiki/index.php/Jetson_Nano/GStreamer)

---

## 🎓 学习曲线

### Phase 1-2 (基础)
- 熟悉多路pipeline管理
- 掌握相机标定流程
- 难度: ⭐⭐

### Phase 3-4 (进阶)
- 理解透视变换原理
- 掌握CUDA图像处理
- 难度: ⭐⭐⭐⭐

### Phase 5-6 (高级)
- CUDA性能优化
- 系统级调优
- 难度: ⭐⭐⭐⭐⭐

---

## 📝 版本历史

- **v0.1** (当前): 单路CUDA处理和RTSP推流 ✅
- **v0.2** (计划): 四路同步采集
- **v0.3** (计划): 畸变校正
- **v0.4** (计划): 透视变换
- **v0.5** (计划): 多视图融合
- **v1.0** (目标): 完整环视系统

---

## 💡 未来扩展

- [ ] 支持车道线检测叠加
- [ ] 支持障碍物检测标注
- [ ] 支持录像功能
- [ ] 支持Web界面控制
- [ ] 支持动态标定
- [ ] 支持3D鸟瞰图

