# 🎯 环视拼接系统 - 开发计划

## 📋 项目目标

实现基于 CUDA 的实时四路摄像头环视拼接系统，支持 RTSP 推流输出。
系统提供 **两种渲染模式**，可根据应用场景切换：

- **2D 鸟瞰图模式**：传统俯视图，使用单应性矩阵（Homography），性能高、延迟低
- **3D 碗状渲染模式**：沉浸式 3D 环视，使用 OpenGL ES，支持视角交互

---

## 🏗️ 系统架构设计

### 2D 模式架构
```
┌─────────────────────────────────────────────────────────────┐
│                      四路摄像头输入                          │
│  Camera 0      Camera 1      Camera 2      Camera 3        │
│  (前视)        (后视)        (左视)        (右视)           │
│    ↓             ↓             ↓             ↓              │
├─────────────────────────────────────────────────────────────┤
│                    V4L2 + nvvidconv                         │
│              NVMM Buffer (四路独立buffer)                   │
│    ↓             ↓             ↓             ↓              │
├─────────────────────────────────────────────────────────────┤
│                  CUDA 2D 拼接处理                           │
│  • 查表法 (LUT) 快速映射                                    │
│  • 基于 Homography 的透视变换                               │
│  • 加权融合 (Alpha Blending)                                │
│    ↓                                                        │
├─────────────────────────────────────────────────────────────┤
│              拼接后的单路输出 (NVMM)                         │
│              2048×2048 鸟瞰图                               │
│    ↓                                                        │
├─────────────────────────────────────────────────────────────┤
│          H.264编码 (nvv4l2h264enc)                          │
│    ↓                                                        │
├─────────────────────────────────────────────────────────────┤
│              RTSP Server 推流                               │
└─────────────────────────────────────────────────────────────┘
```

### 3D 模式架构
```
┌─────────────────────────────────────────────────────────────┐
│                      四路摄像头输入                          │
│    ↓             ↓             ↓             ↓              │
├─────────────────────────────────────────────────────────────┤
│                    V4L2 + nvvidconv                         │
│              NVMM Buffer (四路独立buffer)                   │
│    ↓             ↓             ↓             ↓              │
├─────────────────────────────────────────────────────────────┤
│              OpenGL ES 3D 渲染管线                          │
│  • 3D Bowl Mesh (碗状模型)                                  │
│  • 相机外参 (R, T) 纹理映射                                 │
│  • 深度测试 + 插值                                          │
│  • 可交互视角控制                                           │
│    ↓                                                        │
├─────────────────────────────────────────────────────────────┤
│              渲染后的输出 (FBO)                             │
│              1920×1080 3D 视图                              │
│    ↓                                                        │
├─────────────────────────────────────────────────────────────┤
│          H.264编码 + RTSP 推流                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 📅 开发阶段

### ✅ Phase 0: 基础设施 (已完成)

**目标**: 建立单路处理和推流基础

**完成项**:
- [x] 单路 V4L2 摄像头采集
- [x] NvBufSurface API 集成
- [x] CUDA kernel 基础处理 (亮度调整)
- [x] H.264 硬件编码
- [x] RTSP Server 推流
- [x] Python 版 2D 拼接算法原型 (`stitching/`)
  - [x] 相机标定参数加载
  - [x] 查找表 (LUT) 生成
  - [x] CUDA 拼接 kernel
  - [x] 可视化验证

**经验总结**:
- NVMM 内存需要通过 `NvBufSurfaceMap` 映射
- 当前使用 `cudaMemcpy` (伪零拷贝)，真零拷贝需 EGL Interop
- `identity` element 作为 CUDA 处理 hook 点
- Python 原型验证了算法可行性

---

## 🛣️ 开发路径

项目采用 **双路径并行** 策略：
- **Path A (2D)**: 优先级高，快速交付实用功能
- **Path B (3D)**: 高级功能，提升用户体验

---

## 🅰️ Path A: 2D 鸟瞰图模式 (优先)

### 🚧 Phase A1: 算法移植与集成 (进行中)

**目标**: 将 Python 版 2D 拼接算法移植到 C++ 并集成到推流管线

**任务列表**:
- [x] 定义 `Surrounder` 类和查表结构
- [x] 移植 `surround_kernel` CUDA 核函数
- [x] 实现 `surround_view.binary` 加载逻辑
- [x] 单目模拟四目测试 (分屏效果验证)
- [x] **集成到 GStreamer 管线** 
  - [x] 修改 `main.cpp` 支持双模采集 (Sim/Real)
  - [x] 实现 NVMM Adapter 支持多路 NvBufSurface 处理
  - [x] 实现仿真模式下的 AppSrc 数据循环推流
- [x] **视觉质量优化 (Stitching Optimization)**
  - [x] **亮度/色彩均衡**：实现了参考 `neozhaoliang` 仓库的闭环 BGR 比例优化算法。
  - [x] **边缘羽化**：LUT 生成脚本已支持 80px 宽度的边缘柔化。
  - [x] **平滑融合**：引入 30° 融合带宽与 Sine 权重曲线。
- [ ] **性能测试与极致优化 (Priority 2)**
  - [ ] **真·零拷贝实现 (Current Focus)**：移除 RTSP 推流前的 `cudaMemcpyDeviceToHost`，改用 GPU 编码器直接读取 NVMM。
  - [x] **端到端延迟分析**：已在日志中集成采集到推流的全链路耗时统计。
- [ ] **动态系统演进 (Priority 3)**
  - [ ] **在线标定集成**：支持加载标定算法输出的新外参。
  - [ ] **LUT 实时热更新**：实现无需重启程序的查找表重算与显存重加载逻辑。

**预期输出**:
- 单路 RTSP 流输出 2D 拼接后的鸟瞰图
- 帧率: 30fps @ 1920×1080
- 延迟: < 200ms

---

## 📷 Phase 2: 统一标定 (待开始)

**目标**: 一次性完成四路摄像头的完整标定，获取所有参数供 2D 和 3D 模式使用

**核心思想**: 
- 标定一次，参数通用
- 获取完整的内参 (K, D) 和外参 (R, T)
- 从外参派生 Homography (H) 用于 2D 模式

### 标定参数关系

```
完整标定参数
├── 内参 (K, D)                    ← 相机固有属性
│   ├── camera_matrix (K)          ← 焦距、主点
│   └── dist_coeffs (D)            ← 畸变系数
│
└── 外参 (R, T)                    ← 相机在世界坐标系中的位姿
    ├── Rotation (R)               ← 3×3 旋转矩阵
    └── Translation (T)            ← 3×1 平移向量
    
派生参数
└── Homography (H)                 ← 从 K, R, T 派生，用于 2D 模式
    └── H = K * [r1, r2, t]        ← 假设地面 Z=0
```

**关键公式**:
```
对于地面点 (X, Y, 0):
P_img = K * [R | T] * [X, Y, 0, 1]^T
      = K * [r1, r2, t] * [X, Y, 1]^T
      = H * [X, Y, 1]^T

其中 r1, r2 是 R 的前两列
```

---

### 任务列表

#### 2.1 硬件准备
- [ ] 确认四个摄像头设备路径 (`/dev/video0-3`)
- [ ] 验证每个摄像头的分辨率和帧率 (建议 1920×1080 @ 30fps)
- [ ] 测试 USB/CSI 带宽是否足够支持四路同时采集
- [ ] 确定每路摄像头的物理位置（前/后/左/右）
- [ ] 准备标定工具
  - [ ] 棋盘格标定板 (推荐 9×6, 方格尺寸 25mm)
  - [ ] 地面标定网格 (1m × 1m, 至少 4×4)
  - [ ] 标定图像采集脚本

---

#### 2.2 相机内参标定

**目标**: 获取每个相机的内参矩阵 (K) 和畸变系数 (D)

**步骤**:
1. **采集标定图像**
   - [ ] 每个摄像头采集 20-30 张棋盘格图像
   - [ ] 覆盖不同角度、距离和位置
   - [ ] 确保棋盘格清晰可见，无运动模糊

2. **运行标定**
   - [ ] 使用 OpenCV `calibrateCamera` 函数
   - [ ] 检查重投影误差 (应 < 0.5 像素)
   - [ ] 保存内参到 YAML 文件

**标定脚本示例**:
```python
import cv2
import numpy as np
import glob

# 棋盘格参数
CHECKERBOARD = (9, 6)
square_size = 25  # mm

# 准备物体点
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= square_size

objpoints = []  # 3D 点
imgpoints = []  # 2D 点

images = glob.glob('calibration_images/front/*.jpg')
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
    
    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)

# 标定
ret, K, D, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None
)

print(f"重投影误差: {ret}")
print(f"内参矩阵 K:\n{K}")
print(f"畸变系数 D:\n{D}")
```

**输出格式** (`yaml/front.yaml`):
```yaml
camera_matrix: !!opencv-matrix
   rows: 3
   cols: 3
   dt: d
   data: [fx, 0, cx, 0, fy, cy, 0, 0, 1]
dist_coeffs: !!opencv-matrix
   rows: 4
   cols: 1
   dt: d
   data: [k1, k2, p1, p2]
resolution: [1920, 1080]
```

---

#### 2.3 相机外参标定

**目标**: 获取每个相机在世界坐标系中的位姿 (R, T)

**采用方法**: **棋盘格 3D 标定** (精度高，自动化)

---

**实施步骤**:

**Step 1: 准备标定环境**
- [ ] 使用与内参标定相同的棋盘格 (9×6, 25mm)
- [ ] 在车辆周围选择 3-5 个标定位置
  - 位置 1: 车辆正前方 (距离 1.5-2m)
  - 位置 2: 车辆右前方 (45度角)
  - 位置 3: 车辆右后方
  - 位置 4: 车辆左前方
  - 位置 5: 车辆正后方
- [ ] 每个位置放置棋盘格，确保至少 2 个相机能同时看到
- [ ] 使用三脚架或支架固定棋盘格，保持平整

**Step 2: 定义世界坐标系**
- [ ] 选择车辆中心为世界坐标系原点
- [ ] 定义坐标轴:
  - **X 轴**: 车辆右侧方向 (正方向向右)
  - **Y 轴**: 车辆前进方向 (正方向向前)
  - **Z 轴**: 垂直向上 (正方向向上)
- [ ] 测量并记录每个标定位置的棋盘格中心坐标 (X, Y, Z)

**Step 3: 采集标定图像**
- [ ] 同时采集四路摄像头的图像 (确保时间戳同步)
- [ ] 每个标定位置采集 3-5 张图像 (轻微调整棋盘格角度)
- [ ] 确保棋盘格在图像中清晰可见，无运动模糊
- [ ] 总共采集 15-25 组四路同步图像

**Step 4: 运行标定脚本**

**标定脚本** (`calibration/calibrate_extrinsics.py`):
```python
import cv2
import numpy as np
import glob
import yaml

# 棋盘格参数
CHECKERBOARD = (9, 6)
SQUARE_SIZE = 25  # mm

# 世界坐标系定义
# 棋盘格位置 (相对于车辆中心，单位: cm)
BOARD_POSITIONS = {
    'pos1': np.array([0, 200, 0]),      # 正前方 2m
    'pos2': np.array([150, 150, 0]),    # 右前方
    'pos3': np.array([150, -150, 0]),   # 右后方
    'pos4': np.array([-150, 150, 0]),   # 左前方
    'pos5': np.array([0, -200, 0]),     # 正后方
}

def prepare_object_points():
    """准备棋盘格的 3D 物体点 (相对于棋盘格中心)"""
    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE
    
    # 将原点移到棋盘格中心
    objp[:, 0] -= (CHECKERBOARD[0] - 1) * SQUARE_SIZE / 2
    objp[:, 1] -= (CHECKERBOARD[1] - 1) * SQUARE_SIZE / 2
    
    return objp

def calibrate_camera_extrinsics(camera_name, K, D):
    """
    标定单个相机的外参
    
    Args:
        camera_name: 'front', 'back', 'left', 'right'
        K: 内参矩阵
        D: 畸变系数
    
    Returns:
        R, T: 旋转矩阵和平移向量
    """
    objp_local = prepare_object_points()  # 棋盘格局部坐标
    
    world_points = []
    image_points = []
    
    # 遍历所有标定位置
    for pos_name, board_center in BOARD_POSITIONS.items():
        images = glob.glob(f'calibration_images/{pos_name}/{camera_name}/*.jpg')
        
        for img_path in images:
            img = cv2.imread(img_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # 检测棋盘格角点
            ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
            
            if ret:
                # 亚像素精度优化
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                
                # 将棋盘格局部坐标转换为世界坐标
                # 假设棋盘格平行于地面 (Z=0 平面)
                objp_world = objp_local.copy()
                objp_world[:, 0] += board_center[0]  # X
                objp_world[:, 1] += board_center[1]  # Y
                objp_world[:, 2] += board_center[2]  # Z (通常为 0)
                
                world_points.append(objp_world)
                image_points.append(corners.reshape(-1, 2))
    
    if len(world_points) == 0:
        raise ValueError(f"No valid checkerboard found for {camera_name}")
    
    # 将所有点合并
    world_points = np.vstack(world_points).astype(np.float32)
    image_points = np.vstack(image_points).astype(np.float32)
    
    # 使用 solvePnP 计算外参
    success, rvec, tvec = cv2.solvePnP(
        world_points,
        image_points,
        K,
        D,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    
    if not success:
        raise ValueError(f"solvePnP failed for {camera_name}")
    
    # 转换为旋转矩阵
    R, _ = cv2.Rodrigues(rvec)
    T = tvec
    
    # 计算重投影误差
    projected, _ = cv2.projectPoints(world_points, rvec, tvec, K, D)
    projected = projected.reshape(-1, 2)
    error = np.sqrt(np.mean((image_points - projected) ** 2))
    
    print(f"{camera_name} 外参标定完成:")
    print(f"  重投影误差: {error:.3f} 像素")
    print(f"  使用点数: {len(world_points)}")
    
    return R, T, error

def save_extrinsics(camera_name, R, T, output_dir='yaml'):
    """保存外参到 YAML 文件"""
    yaml_path = f'{output_dir}/{camera_name}.yaml'
    
    # 读取现有的 YAML (包含内参)
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    
    # 添加外参
    data['extrinsics'] = {
        'rotation': {
            'rows': 3,
            'cols': 3,
            'dt': 'f',
            'data': R.flatten().tolist()
        },
        'translation': {
            'rows': 3,
            'cols': 1,
            'dt': 'd',
            'data': T.flatten().tolist()
        }
    }
    
    # 保存
    with open(yaml_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)
    
    print(f"外参已保存到 {yaml_path}")

# 主程序
if __name__ == "__main__":
    cameras = ['front', 'back', 'left', 'right']
    
    for cam in cameras:
        # 加载内参
        yaml_path = f'yaml/{cam}.yaml'
        fs = cv2.FileStorage(yaml_path, cv2.FILE_STORAGE_READ)
        K = fs.getNode("camera_matrix").mat()
        D = fs.getNode("dist_coeffs").mat()
        fs.release()
        
        # 标定外参
        R, T, error = calibrate_camera_extrinsics(cam, K, D)
        
        # 保存
        save_extrinsics(cam, R, T)
    
    print("\n所有相机外参标定完成!")
```

**Step 5: 验证标定精度**
- [ ] 检查重投影误差 (应 < 1.0 像素)
- [ ] 可视化相机位姿 (3D 坐标系)
- [ ] 验证相机朝向是否合理

**验证脚本** (`calibration/visualize_extrinsics.py`):
```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2

def visualize_camera_poses(yaml_dir='yaml'):
    """可视化四个相机的位姿"""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    cameras = ['front', 'back', 'left', 'right']
    colors = ['red', 'blue', 'green', 'orange']
    
    for cam, color in zip(cameras, colors):
        # 加载外参
        fs = cv2.FileStorage(f'{yaml_dir}/{cam}.yaml', cv2.FILE_STORAGE_READ)
        R = fs.getNode("extrinsics").getNode("rotation").mat()
        T = fs.getNode("extrinsics").getNode("translation").mat()
        fs.release()
        
        # 相机中心在世界坐标系中的位置
        C = -R.T @ T
        
        # 相机朝向 (Z 轴方向)
        Z_axis = R.T @ np.array([[0], [0], [1]])
        
        # 绘制相机位置
        ax.scatter(C[0], C[1], C[2], c=color, s=100, label=cam)
        
        # 绘制相机朝向
        ax.quiver(C[0], C[1], C[2], 
                  Z_axis[0], Z_axis[1], Z_axis[2],
                  length=50, color=color, arrow_length_ratio=0.3)
    
    # 绘制车辆 (简化为矩形)
    car_length = 400  # cm
    car_width = 180
    car_corners = np.array([
        [-car_width/2, -car_length/2, 0],
        [car_width/2, -car_length/2, 0],
        [car_width/2, car_length/2, 0],
        [-car_width/2, car_length/2, 0],
        [-car_width/2, -car_length/2, 0],
    ])
    ax.plot(car_corners[:, 0], car_corners[:, 1], car_corners[:, 2], 'k-', linewidth=2)
    
    ax.set_xlabel('X (cm)')
    ax.set_ylabel('Y (cm)')
    ax.set_zlabel('Z (cm)')
    ax.legend()
    ax.set_title('Camera Extrinsics Visualization')
    plt.show()

if __name__ == "__main__":
    visualize_camera_poses()
```

---

**备选方案**: **地面点标定** (如果棋盘格标定困难)

如果棋盘格放置困难或精度不满足要求，可以使用地面点标定作为备选:
- 在车辆周围地面绘制已知坐标的标记点 (如 ArUco 标记)
- 手动标注每个相机图像中的地面点
- 使用 `cv2.solvePnP` 计算 R 和 T

详细步骤见原 TODO.md 方案 2。

---

**输出格式** (`yaml/front.yaml` 追加):
```yaml
# ... 内参部分 ...
extrinsics:
  rotation: !!opencv-matrix
     rows: 3
     cols: 3
     dt: f
     data: [r11, r12, r13, r21, r22, r23, r31, r32, r33]
  translation: !!opencv-matrix
     rows: 3
     cols: 1
     dt: d
     data: [tx, ty, tz]
```

---

#### 2.4 派生 Homography 矩阵 (用于 2D 模式)

**目标**: 从外参 (R, T) 计算地面投影的 Homography 矩阵 (H)

**原理**:
对于地面点 (Z=0)，投影简化为:
```
H = K * [r1, r2, t]
```
其中 r1, r2 是 R 的第 1、2 列。

**计算脚本**:
```python
import numpy as np

def compute_homography_from_extrinsics(K, R, T):
    """
    从内参和外参计算地面 Homography
    
    Args:
        K: 3×3 内参矩阵
        R: 3×3 旋转矩阵
        T: 3×1 平移向量
    
    Returns:
        H: 3×3 Homography 矩阵
    """
    # 提取 R 的前两列
    r1 = R[:, 0]
    r2 = R[:, 1]
    
    # 构造 [r1, r2, t]
    RT = np.column_stack((r1, r2, T.flatten()))
    
    # H = K * [r1, r2, t]
    H = K @ RT
    
    # 归一化 (使 H[2,2] = 1)
    H = H / H[2, 2]
    
    return H

# 示例
H_front = compute_homography_from_extrinsics(K_front, R_front, T_front)
print(f"Homography 矩阵 H:\n{H_front}")
```

**输出格式** (`yaml/front.yaml` 追加):
```yaml
# ... 内参和外参部分 ...
project_matrix: !!opencv-matrix
   rows: 3
   cols: 3
   dt: d
   data: [h11, h12, h13, h21, h22, h23, h31, h32, h33]
```

---

#### 2.5 生成查找表 (用于 2D 模式)

**目标**: 从 Homography 生成 CUDA 拼接所需的查找表

**步骤**:
- [ ] 修改 `stitching/generate_data.py` 以支持从 YAML 加载参数
- [ ] 运行脚本生成 `surround_view.binary`
- [ ] 可视化验证拼接效果
- [ ] 将 `.binary` 文件部署到 Jetson

**生成命令**:
```bash
cd stitching
python generate_data.py --config yaml/
```

---

### 预期输出

**标定参数文件** (`yaml/` 目录):
```
yaml/
├── front.yaml        # 前视相机完整参数
├── back.yaml         # 后视相机完整参数
├── left.yaml         # 左视相机完整参数
└── right.yaml        # 右视相机完整参数
```

**每个 YAML 文件包含**:
- ✅ 内参 (camera_matrix, dist_coeffs)
- ✅ 外参 (rotation, translation)
- ✅ Homography (project_matrix, 从外参派生)
- ✅ 分辨率 (resolution)

**查找表文件** (用于 2D 模式):
- `surround_view.binary` (约 76MB @ 1920×1080 输出)

**标定报告**:
- 重投影误差统计
- 外参可视化 (相机位姿 3D 图)
- 拼接效果预览图

---

## 🅰️ Path A: 2D 鸟瞰图模式 (优先)

### 🚀 Phase A3: 四路采集与推流 (待开始)

**前置条件**: Phase 2 标定完成

**目标**: 实现真实四路摄像头的同步采集和 2D 拼接推流

**任务列表**:

#### A3.1 多路 Pipeline 架构
- [ ] 设计四路独立 GStreamer pipeline
- [ ] 实现 pipeline 同步机制（时间戳对齐）
- [ ] 创建四路 buffer 管理器
- [ ] 实现帧同步检测（确保四路帧时间戳一致，误差 < 33ms）

**代码结构**:
```
cpp_demo/src/
├── main.cpp                          # 主程序
├── camera_manager.h/cpp              # 摄像头管理器
├── multi_pipeline.h/cpp              # 多路pipeline管理
├── frame_sync.h/cpp                  # 帧同步器
└── cuda/
    ├── surrounder.h/cu               # 2D拼接处理器
    └── nvbuffer_utils.h/cu           # NVMM工具函数
```

#### A3.2 集成测试
- [ ] 四路视频成功同步采集
- [ ] 2D 拼接输出正确
- [ ] RTSP 推流稳定
- [ ] 帧率稳定在 30fps
- [ ] 端到端延迟 < 200ms

**预期输出**:
- 完整的 2D 环视系统
- RTSP 流地址: `rtsp://<JETSON_IP>:8554/surround_2d`

---

### ⚡ Phase A4: 性能优化 (待开始)

**目标**: 优化管线和算法以降低延迟和资源占用

**任务列表**:
- [ ] 实现真零拷贝 (EGL Interop)
- [ ] 优化 CUDA Kernel
  - [ ] 使用 Texture Memory 优化随机访问
  - [ ] 向量化读取 (uchar4)
  - [ ] 合并 kernel 减少启动开销
- [ ] Pipeline 优化
  - [ ] 使用 CUDA Streams 并发处理
  - [ ] Double buffering 避免等待
- [ ] 降低分辨率选项 (720p 模式)

**性能指标**:
| 阶段 | 目标延迟 | 实际延迟 |
|------|---------|---------| 
| 四路采集 | < 10ms | TBD |
| CUDA 拼接 | < 20ms | TBD |
| H.264 编码 | < 15ms | TBD |
| **总延迟** | **< 50ms** | **TBD** |

---

## 🅱️ Path B: 3D 碗状渲染模式 (高级)

### 🎨 Phase B2: 3D Mesh 生成 (待开始)

**前置条件**: Phase 2 标定完成 (已有外参 R, T)

**目标**: 生成碗状 3D 模型用于纹理映射

**任务列表**:
- [ ] 实现 Bowl Mesh 生成算法
  - [ ] 参考 SokratG/Surround-View 的 `Bowl.cpp`
  - [ ] 定义碗的几何参数 (半径、高度、曲率)
- [ ] 计算 Mesh 顶点的 UV 坐标
  - [ ] 使用外参 (R, T) 将 3D 点投影到各相机
  - [ ] 根据象限分配相机索引
- [ ] 生成 Mesh 数据文件
  - [ ] 顶点坐标 (X, Y, Z)
  - [ ] UV 坐标 (u, v)
  - [ ] 相机索引 (cam_id)
  - [ ] 三角形索引

**输出格式**:
```
surround_view_3d.mesh (二进制)
- Header: Magic, NumVerts, NumIndices
- Vertices: float[NumVerts * 3]
- UVs: float[NumVerts * 2]
- CamIndices: int[NumVerts]
- Indices: uint[NumIndices]
```

---

### 🖼️ Phase B3: OpenGL ES 渲染管线 (待开始)

**目标**: 使用 OpenGL ES 3.2 实现 3D 渲染

**任务列表**:

#### B3.1 EGL 上下文初始化
- [ ] 创建 Headless EGL Display (无窗口渲染)
- [ ] 配置 EGL Context (OpenGL ES 3.2)
- [ ] 创建 FBO (Framebuffer Object) 用于离屏渲染

#### B3.2 Shader 编写
**Vertex Shader**:
```glsl
#version 320 es
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec2 aTexCoord;
layout(location = 2) in int aCamIndex;

uniform mat4 uModel;
uniform mat4 uView;
uniform mat4 uProjection;

out vec2 TexCoord;
flat out int CamIndex;

void main() {
    gl_Position = uProjection * uView * uModel * vec4(aPos, 1.0);
    TexCoord = aTexCoord;
    CamIndex = aCamIndex;
}
```

**Fragment Shader**:
```glsl
#version 320 es
precision mediump float;

in vec2 TexCoord;
flat in int CamIndex;

uniform sampler2D uTextures[4];

out vec4 FragColor;

void main() {
    if (CamIndex == 0) FragColor = texture(uTextures[0], TexCoord);
    else if (CamIndex == 1) FragColor = texture(uTextures[1], TexCoord);
    else if (CamIndex == 2) FragColor = texture(uTextures[2], TexCoord);
    else if (CamIndex == 3) FragColor = texture(uTextures[3], TexCoord);
    else FragColor = vec4(0.0, 0.0, 0.0, 1.0);
}
```

#### B3.3 渲染循环
- [ ] 加载 Mesh 数据到 VBO/VAO
- [ ] 上传四路摄像头纹理到 GPU
- [ ] 设置 View 和 Projection 矩阵
- [ ] 渲染到 FBO
- [ ] 读取 FBO 像素数据到 NVMM Buffer
- [ ] 送入 H.264 编码器

#### B3.4 NVMM-OpenGL 互操作
- [ ] 使用 EGLImage 实现 NVMM 和 OpenGL 纹理共享
- [ ] 避免 CPU-GPU 拷贝

**参考资料**:
- [Jetson EGLImage Extension](https://docs.nvidia.com/jetson/l4t-multimedia/group__ee__nvbuffering__group.html)
- [OpenGL ES 3.2 Spec](https://www.khronos.org/registry/OpenGL/specs/es/3.2/es_spec_3.2.pdf)

---

### 🎮 Phase B4: 交互控制 (可选)

**目标**: 支持实时视角调整

**任务列表**:
- [ ] 实现相机位置控制 (Eye Position)
- [ ] 实现相机朝向控制 (LookAt Target)
- [ ] 通过 RTSP 元数据或 HTTP API 接收控制指令
- [ ] 实时更新 View 矩阵

---

## 🔧 技术选型

### 核心库
- **GStreamer 1.x**: 视频 pipeline 管理
- **CUDA 11.4**: GPU 加速计算 (2D 模式)
- **OpenGL ES 3.2**: 3D 渲染 (3D 模式)
- **EGL**: Headless 渲染上下文
- **NvBuffer API**: NVMM 内存管理
- **yaml-cpp**: 配置文件解析

### 硬件要求
- **平台**: NVIDIA Jetson Orin / Xavier
- **摄像头**: 4×USB/CSI 摄像头 (推荐 960×640 @ 30fps)
- **内存**: 建议 ≥ 8GB
- **存储**: ≥ 16GB (用于标定数据和模型)

---

## 📊 性能目标

### 2D 模式
| 指标 | 目标值 |
|------|--------|
| 输入分辨率 | 4 × 960×640 @ 30fps |
| 输出分辨率 | 2048×2048 @ 30fps |
| 端到端延迟 | < 50ms |
| GPU 占用率 | < 60% |
| 功耗 | < 20W |

### 3D 模式
| 指标 | 目标值 |
|------|--------|
| 输入分辨率 | 4 × 960×640 @ 30fps |
| 输出分辨率 | 1920×1080 @ 30fps |
| 端到端延迟 | < 100ms |
| GPU 占用率 | < 80% |
| 功耗 | < 25W |

---

## 🔍 风险与挑战

### 技术风险

#### 2D 模式
1. **多路同步**: 四路摄像头时间戳对齐困难
   - **缓解**: 使用硬件时间戳，容忍小误差 (< 1 帧)

2. **标定精度**: Homography 标定误差导致拼接错位
   - **缓解**: 使用高精度标定板，多次标定取平均

#### 3D 模式
1. **外参标定**: 从 Homography 恢复外参可能不准确
   - **缓解**: 使用 3D 标定板 (如 AprilTag 3D) 直接标定外参

2. **OpenGL 性能**: 3D 渲染可能无法达到 30fps
   - **缓解**: 降低 Mesh 分辨率，优化 Shader

3. **NVMM-OpenGL 互操作**: EGLImage 集成复杂
   - **缓解**: 先使用 CPU 拷贝验证逻辑，后期优化

### 硬件风险
1. **USB 带宽限制**: USB 3.0 可能无法支持 4×960p
   - **缓解**: 使用 CSI 接口，或降低部分摄像头分辨率

2. **摄像头质量**: 不同摄像头色彩不一致
   - **缓解**: 添加颜色校准步骤 (Histogram Matching)

---

## 📚 参考资料

### 环视拼接算法
- [Around View Monitor (AVM) System](https://en.wikipedia.org/wiki/Around_view_monitor)
- [OpenCV Camera Calibration](https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html)
- [SokratG/Surround-View](https://github.com/SokratG/Surround-View) - 3D 碗状渲染参考

### CUDA 优化
- [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)

### OpenGL ES
- [OpenGL ES 3.2 Specification](https://www.khronos.org/registry/OpenGL/specs/es/3.2/es_spec_3.2.pdf)
- [Learn OpenGL ES](https://learnopengl.com/)

### Jetson 多媒体
- [Jetson Linux Multimedia API](https://docs.nvidia.com/jetson/l4t-multimedia/index.html)
- [GStreamer on Jetson](https://developer.ridgerun.com/wiki/index.php/Jetson_Nano/GStreamer)
- [EGLImage Extension](https://www.khronos.org/registry/EGL/extensions/KHR/EGL_KHR_image.txt)

---

## 📝 版本历史

- **v0.1** (已完成): 单路 CUDA 处理和 RTSP 推流 ✅
- **v0.2** (已完成): Python 版 2D 拼接算法原型 ✅
- **v0.3** (进行中): C++ 版 2D 拼接算法移植 🚧
- **v0.4** (计划): 实车标定与四路采集
- **v0.5** (计划): 2D 模式完整系统
- **v1.0** (目标): 2D + 3D 双模式系统

---

## 💡 未来扩展

### 功能扩展
- [ ] 支持车道线检测叠加
- [ ] 支持障碍物检测标注
- [ ] 支持录像功能 (MP4 文件)
- [ ] 支持 Web 界面控制
- [ ] 支持动态标定 (在线校准)
- [ ] 透明底盘 (历史帧补偿渲染)

### 算法增强
- [ ] 多频段融合 (Multi-band Blending)
- [ ] 泊松融合 (Poisson Blending)
- [ ] 自动曝光平衡
- [ ] 运动模糊补偿

---

## 🎓 学习曲线

### Path A (2D 模式)
- **Phase A1-A2**: 熟悉标定流程和参数格式 - 难度: ⭐⭐
- **Phase A3**: 多路 pipeline 管理和同步 - 难度: ⭐⭐⭐
- **Phase A4**: CUDA 性能优化 - 难度: ⭐⭐⭐⭐

### Path B (3D 模式)
- **Phase B1**: 理解外参标定原理 - 难度: ⭐⭐⭐
- **Phase B2**: 3D 几何和纹理映射 - 难度: ⭐⭐⭐⭐
- **Phase B3**: OpenGL ES 渲染管线 - 难度: ⭐⭐⭐⭐⭐
- **Phase B4**: 系统集成和优化 - 难度: ⭐⭐⭐⭐⭐

---

## 🚦 当前状态

**✅ 已完成**:
- Phase 0: 基础设施
- Python 版 2D 拼接算法

**🚧 进行中**:
- Phase A1: 真·零拷贝性能优化 (Zero-copy)

**📋 下一步**:
- Phase A1: 集成到 GStreamer 管线
- Phase A2: 实车标定

**🎯 短期目标 (1-2 周)**:
- 完成 Phase A1
- 开始 Phase A2 硬件准备

**🎯 中期目标 (1-2 月)**:
- 完成 2D 模式 (Phase A2-A4)
- 交付可用的 2D 环视系统

**🎯 长期目标 (3-6 月)**:
- 完成 3D 模式 (Phase B1-B4)
- 实现双模式切换
