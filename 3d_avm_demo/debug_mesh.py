#!/usr/bin/env python3
import json
import numpy as np
import cv2
import os
from scipy.spatial.transform import Rotation as SciRot

# ==========================================
# PART 1: 核心类 (精简版)
# ==========================================
def ensure_point_list(points, dim, concatenate=True, crop=True):
    if isinstance(points, list):
        points = np.array(points)
    if crop:
        for test_dim in range(4, dim, -1):
            if points.shape[1] == test_dim:
                points = points[:, 0:test_dim-1]
    if concatenate and points.shape[1] == (dim - 1):
        points = np.concatenate((np.array(points), np.ones((points.shape[0], 1))), axis=1)
    return points

class RadialPolyCamProjection:
    def __init__(self, distortion_params):
        self.coefficients = np.asarray(distortion_params)
        self.power = np.array([np.arange(start=1, stop=self.coefficients.size + 1)]).T
            
    def project_3d_to_2d(self, cam_points):
        camera_points = ensure_point_list(cam_points, dim=3)
        chi = np.sqrt(camera_points.T[0]**2 + camera_points.T[1]**2)
        theta = np.pi / 2.0 - np.arctan2(camera_points.T[2], chi)
        rho = np.dot(self.coefficients, np.power(np.array([theta]), self.power))
        
        with np.errstate(divide='ignore', invalid='ignore'):
            scale = np.divide(rho, chi)
            scale[chi == 0] = 0
        return scale[:, np.newaxis] * camera_points[:, 0:2]

class Camera:
    def __init__(self, lens, translation, rotation, size, aspect_ratio=1.0):
        self.lens = lens
        pose = np.eye(4)
        pose[0:3, 3] = translation
        pose[0:3, 0:3] = rotation
        self._inv_pose = np.linalg.inv(pose)
        self._size = size
        self._principle_point = 0.5 * np.array(size) + np.array([0,0]) - 0.5 # 简化PP
        self._aspect_ratio = np.array([1, aspect_ratio])

    def project_3d_to_2d(self, world_points):
        world_points = ensure_point_list(world_points, dim=4)
        camera_points = world_points @ self._inv_pose.T
        lens_points = self.lens.project_3d_to_2d(camera_points[:, 0:3])
        return (lens_points * self._aspect_ratio) + self._principle_point

def read_cam(path):
    with open(path) as f: config = json.load(f)
    intr = config['intrinsic']
    # 原始四元数顺序 (Woodscape默认)
    rot = SciRot.from_quat(config['extrinsic']['quaternion']).as_matrix()
    lens = RadialPolyCamProjection([intr['k1'], intr['k2'], intr['k3'], intr['k4']])
    return Camera(lens, config['extrinsic']['translation'], rot, (intr['width'], intr['height']), intr['aspect_ratio'])

# ==========================================
# PART 2: 网格逆向可视化工具
# ==========================================

def run_debug_wireframe():
    output_size = (1000, 1000)
    world_range = 12.0
    inner_radius = 1.7 # 你的挖空半径
    
    # 只加载 MVL (左视相机)
    json_path = './calib/cam1.json'
    img_path = './imgs_raw/cam1.png'
    
    if not os.path.exists(json_path): return
    cam = read_cam(json_path)
    img = cv2.imread(img_path)
    
    # 1. 生成稀疏的 3D 网格 (方便看清点)
    # 我们只生成左侧区域的点 (90度方向)
    step = 20 # 步长越大，点越稀疏
    x = np.linspace(world_range, -world_range, output_size[0] // step)
    y = np.linspace(world_range, -world_range, output_size[1] // step)
    grid_x, grid_y = np.meshgrid(x, y, indexing='ij')
    
    # 筛选只在左侧的点 (y > 0) 且在挖空圈外
    radius = np.sqrt(grid_x**2 + grid_y**2)
    mask = (radius > inner_radius) & (grid_y > 0) # 只看左半边
    
    flat_x = grid_x[mask]
    flat_y = grid_y[mask]
    
    # 简单的碗状模型计算 Z
    flat_radius = world_range * 0.45
    z_vals = np.zeros_like(flat_x)
    bowl_mask = radius[mask] > flat_radius
    z_vals[bowl_mask] = (radius[mask][bowl_mask] - flat_radius)**2 * 0.20
    
    world_points = np.stack([flat_x, flat_y, z_vals], axis=1)
    
    # 2. 投影回原图
    uv = cam.project_3d_to_2d(world_points)
    
    # 3. 在原图上画点
    debug_img = img.copy()
    count_valid = 0
    count_total = len(uv)
    
    for point in uv:
        u, v = int(point[0]), int(point[1])
        # 检查点是否在图像内
        if 0 <= u < cam._size[0] and 0 <= v < cam._size[1]:
            # 画绿色小圆点
            cv2.circle(debug_img, (u, v), 2, (0, 255, 0), -1)
            count_valid += 1
        else:
            pass # 点落在了图像外
            
    # 画一个红色的圈表示本来想挖空的位置 (近似)
    # 这只是个示意，因为投影是畸变的
    
    print(f"总投射点数: {count_total}")
    print(f"落在图像内的点数: {count_valid}")
    print(f"图像外丢失点数: {count_total - count_valid}")
    
    cv2.imwrite("debug_mesh_on_cam1.jpg", debug_img)
    print("已保存: debug_mesh_on_cam1.jpg")
    print("请打开这张图：")
    print("1. 绿点是否覆盖了你想保留的黑色影子区域？")
    print("2. 如果绿点到了那个区域就戛然而止，说明是FOV不够或投影越界。")

if __name__ == "__main__":
    run_debug_wireframe()