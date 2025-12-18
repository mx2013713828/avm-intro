#!/usr/bin/env python3
import json
import numpy as np
import cv2
import os
from scipy.spatial.transform import Rotation as SciRot

# ==========================================
# PART 1: 官方核心类 (保持不变)
# ==========================================
def ensure_point_list(points, dim, concatenate=True, crop=True):
    if isinstance(points, list):
        points = np.array(points)
    assert isinstance(points, np.ndarray)
    assert points.ndim == 2
    if crop:
        for test_dim in range(4, dim, -1):
            if points.shape[1] == test_dim:
                new_shape = test_dim - 1
                assert np.array_equal(points[:, new_shape], np.ones(points.shape[0]))
                points = points[:, 0:new_shape]
    if concatenate and points.shape[1] == (dim - 1):
        points = np.concatenate((np.array(points), np.ones((points.shape[0], 1))), axis=1)
    return points

class Projection(object):
    def project_3d_to_2d(self, cam_points: np.ndarray, invalid_value=np.nan):
        raise NotImplementedError()

class RadialPolyCamProjection(Projection):
    def __init__(self, distortion_params: list):
        self.coefficients = np.asarray(distortion_params)
        self.power = np.array([np.arange(start=1, stop=self.coefficients.size + 1)]).T

    def project_3d_to_2d(self, cam_points, invalid_value=np.nan):
        camera_points = ensure_point_list(cam_points, dim=3)
        chi = np.sqrt(camera_points.T[0] * camera_points.T[0] + camera_points.T[1] * camera_points.T[1])
        theta = np.pi / 2.0 - np.arctan2(camera_points.T[2], chi)
        rho = self._theta_to_rho(theta)
        lens_points = np.divide(rho, chi, where=(chi != 0))[:, np.newaxis] * camera_points[:, 0:2]
        lens_points[(chi == 0) & (cam_points[:, 2] == 0)] = invalid_value
        return lens_points

    def _theta_to_rho(self, theta):
        return np.dot(self.coefficients, np.power(np.array([theta]), self.power))

class Camera(object):
    def __init__(self, lens: Projection, translation, rotation, size, principle_point, aspect_ratio: float = 1.0):
        self.lens = lens
        pose = np.eye(4)
        pose[0:3, 3] = translation
        pose[0:3, 0:3] = rotation
        self._pose = np.asarray(pose, dtype=float)
        self._inv_pose = np.linalg.inv(self._pose)
        self._size = np.array([size[0], size[1]], dtype=int)
        self._principle_point = 0.5 * self._size + np.array([principle_point[0], principle_point[1]], dtype=float) - 0.5
        self._aspect_ratio = np.array([1, aspect_ratio], dtype=float)

    size = property(lambda self: self._size)
    width = property(lambda self: self._size[0])
    height = property(lambda self: self._size[1])

    def project_3d_to_2d(self, world_points: np.ndarray, do_clip=False, invalid_value=np.nan):
        world_points = ensure_point_list(world_points, dim=4)
        camera_points = world_points @ self._inv_pose.T
        lens_points = self.lens.project_3d_to_2d(camera_points[:, 0:3], invalid_value=invalid_value)
        screen_points = (lens_points * self._aspect_ratio) + self._principle_point
        return self._apply_clip(screen_points, screen_points) if do_clip else screen_points

    def _apply_clip(self, points, clip_source) -> np.ndarray:
        # 禁用 Clip
        mask = (clip_source[:, 0] < -5000) | (clip_source[:, 0] >= self._size[0]+5000)
        points[mask] = [np.nan]
        return points

def read_cam_from_json(path, force_name=None):
    with open(path) as f:
        config = json.load(f)
    intr = config['intrinsic']
    coefficients = [intr['k1'], intr['k2'], intr['k3'], intr['k4']]
    cam = Camera(
        rotation=SciRot.from_quat(config['extrinsic']['quaternion']).as_matrix(),
        translation=config['extrinsic']['translation'],
        lens=RadialPolyCamProjection(coefficients),
        size=(intr['width'], intr['height']),
        principle_point=(intr['cx_offset'], intr['cy_offset']),
        aspect_ratio=intr['aspect_ratio']
    )
    cam.name = force_name if force_name else config.get('name', 'Unknown')
    return cam

# ==========================================
# PART 2: V12 边界距离场融合 (Distance Field)
# ==========================================

def generate_steep_bowl(bev_size, world_range, inner_radius=1.8):
    """
    几何模型：保持 V3 陡峭参数。
    """
    x = np.linspace(world_range, -world_range, bev_size[0])
    y = np.linspace(world_range, -world_range, bev_size[1])
    grid_x, grid_y = np.meshgrid(x, y, indexing='ij') 
    radius = np.sqrt(grid_x**2 + grid_y**2)
    
    flat_radius = world_range * 0.45  
    bowl_curvature = 0.20             
    
    z_grid = np.zeros_like(radius)
    mask_bowl = radius > flat_radius
    z_grid[mask_bowl] = (radius[mask_bowl] - flat_radius)**2 * bowl_curvature
    
    mask_ego_car = radius < inner_radius
    
    flat_x = grid_x.flatten()
    flat_y = grid_y.flatten()
    flat_z = z_grid.flatten()
    
    flat_x[mask_ego_car.flatten()] = np.nan
    flat_y[mask_ego_car.flatten()] = np.nan
    flat_z[mask_ego_car.flatten()] = np.nan
    
    world_points = np.stack([flat_x, flat_y, flat_z], axis=1)
    
    return world_points, mask_ego_car

def get_border_distance_weight(map_x, map_y, width, height, cam_name):
    """
    【V12 核心算法】：计算像素到最近图像边界的距离。
    """
    # 1. 基础有效性 (Pixel Validity)
    # 预留 1 像素缓冲
    is_inside = (map_x >= 1) & (map_x < width - 1) & \
                (map_y >= 1) & (map_y < height - 1)
    
    # 2. 计算到四个边界的距离 (Distance to Borders)
    dist_l = map_x
    dist_r = width - 1 - map_x
    dist_t = map_y
    dist_b = height - 1 - map_y
    
    # --- 特殊处理侧视相机 ---
    # 如果是 MVL/MVR，我们认为底部边界是“无效”的（因为有车身）
    # 所以我们人为地减小到底部边界的距离，让权重在底部迅速归零
    if 'MV' in cam_name:
        # 裁剪底部 15%
        crop_height = height * 0.15
        # 让 dist_b 变小 (甚至为负)
        dist_b = dist_b - crop_height
    
    # 取最小距离 (Min Distance is the bottleneck)
    # 只要靠近任何一个边界，权重就会降低
    min_dist = np.minimum(np.minimum(dist_l, dist_r), np.minimum(dist_t, dist_b))
    
    # 3. 权重映射
    # 距离小于 0 (在裁剪区外) -> 权重 0
    # 距离越大 -> 权重越大
    weight = np.maximum(0, min_dist)
    
    # 4. 锐化 (Sharpening)
    # 使用平方或三次方，让内部区域 (距离远) 的权重呈指数级增长
    # 这确保了非边缘区域几乎完全由原图主导
    weight = np.power(weight, 2.0)
    
    # 应用 Mask
    final_weight = weight * is_inside.astype(np.float32)
    final_weight[np.isnan(map_x)] = 0
    
    return final_weight

def run_avm_v12_distance_field():
    output_size = (1000, 1000) 
    world_range = 12.0 
    inner_car_radius = 1.7 
    
    cam_configs = [
        {'name': 'FV',  'json': './calib/cam3.json', 'img': './imgs_raw/cam3.png'}, 
        {'name': 'RV',  'json': './calib/cam0.json', 'img': './imgs_raw/cam0.png'}, 
        {'name': 'MVL', 'json': './calib/cam1.json', 'img': './imgs_raw/cam1.png'}, 
        {'name': 'MVR', 'json': './calib/cam2.json', 'img': './imgs_raw/cam2.png'}  
    ]
    
    cameras = []
    images = []
    print("加载数据...")
    for cfg in cam_configs:
        if not os.path.exists(cfg['json']): continue
        cam = read_cam_from_json(cfg['json'], force_name=cfg['name'])
        img = cv2.imread(cfg['img'])
        cameras.append(cam)
        images.append(img)
    if not cameras: return

    print("生成 3D 模型...")
    world_points, mask_ego_car = generate_steep_bowl(output_size, world_range, inner_radius=inner_car_radius)
    
    canvas = np.zeros((output_size[0], output_size[1], 3), dtype=np.float32)
    total_weights = np.zeros((output_size[0], output_size[1]), dtype=np.float32)

    print("V12 边界距离场融合 (Distance Field Blending)...")
    for i, cam in enumerate(cameras):
        # 3D -> 2D
        uv = cam.project_3d_to_2d(world_points)
        map_x = uv[:, 0].reshape(output_size).astype(np.float32)
        map_y = uv[:, 1].reshape(output_size).astype(np.float32)
        
        # 使用边界距离权重
        weight = get_border_distance_weight(map_x, map_y, cam.width, cam.height, cam.name)
        
        img = images[i]
        warped = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
        
        canvas += warped.astype(np.float32) * weight[:, :, np.newaxis]
        total_weights += weight

    # 归一化
    # 只要 weight > 0 就显示 (哪怕是边缘)
    valid_mask = total_weights > 0.0001
    total_weights[~valid_mask] = 1.0 
    
    final_image = canvas / total_weights[:, :, np.newaxis]
    final_image = np.clip(final_image, 0, 255).astype(np.uint8)
    
    final_image[mask_ego_car] = [0, 0, 0] 
    
    cx, cy = output_size[0]//2, output_size[1]//2
    pixels_per_meter = (output_size[0]/2) / world_range
    car_w = int(0.85 * pixels_per_meter) 
    car_h = int(1.9 * pixels_per_meter)
    cv2.rectangle(final_image, (cy - car_w, cx - car_h), (cy + car_w, cx + car_h), (40, 40, 40), 2)
    
    output_file = "woodscape_avm_v12_distance.jpg"
    cv2.imwrite(output_file, final_image)
    print(f"完成! 结果保存在: {output_file}")

if __name__ == "__main__":
    run_avm_v12_distance_field()