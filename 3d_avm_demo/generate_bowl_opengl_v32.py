#!/usr/bin/env python3
import json
import numpy as np
import cv2
import os
import glfw
from OpenGL.GL import *
from OpenGL.GLU import *
from scipy.spatial.transform import Rotation as SciRot

# ==========================================
# PART 1: 核心投影类
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
        term_powers = np.power(theta[:, np.newaxis], self.power.T)
        rho = np.dot(term_powers, self.coefficients)
        with np.errstate(divide='ignore', invalid='ignore'):
            scale = np.divide(rho, chi)
            scale[chi == 0] = 0
        return scale[:, np.newaxis] * camera_points[:, 0:2]

class Camera:
    def __init__(self, lens, translation, rotation, size, aspect_ratio=1.0, name="Unknown"):
        self.lens = lens
        pose = np.eye(4)
        pose[0:3, 3] = translation
        pose[0:3, 0:3] = rotation
        self._inv_pose = np.linalg.inv(pose)
        self._size = size
        self.width = size[0]
        self.height = size[1]
        self._principle_point = 0.5 * np.array(size) + np.array([0,0]) - 0.5 
        self._aspect_ratio = np.array([1, aspect_ratio])
        self.name = name

    def project_3d_to_2d(self, world_points):
        world_points = ensure_point_list(world_points, dim=4)
        camera_points = world_points @ self._inv_pose.T
        lens_points = self.lens.project_3d_to_2d(camera_points[:, 0:3])
        return (lens_points * self._aspect_ratio) + self._principle_point

def read_cam(path, force_name):
    with open(path) as f: config = json.load(f)
    intr = config['intrinsic']
    rot = SciRot.from_quat(config['extrinsic']['quaternion']).as_matrix()
    lens = RadialPolyCamProjection([intr['k1'], intr['k2'], intr['k3'], intr['k4']])
    return Camera(lens, config['extrinsic']['translation'], rot, (intr['width'], intr['height']), intr['aspect_ratio'], name=force_name)

def get_border_distance_weight(map_x, map_y, width, height):
    is_inside = (map_x >= 0) & (map_x < width) & \
                (map_y >= 0) & (map_y < height)
    dist_l = map_x
    dist_r = width - 1 - map_x
    dist_t = map_y
    dist_b = height - 1 - map_y
    min_dist = np.minimum(np.minimum(dist_l, dist_r), np.minimum(dist_t, dist_b))
    norm_dist = np.clip(min_dist / (width / 4.0), 0.0, 1.0)
    weight = np.power(norm_dist, 3.0)
    final_weight = weight * is_inside.astype(np.float32)
    final_weight[np.isnan(map_x)] = 0
    return final_weight

# ==========================================
# PART 2: 生成纹理 (修复: 扩大范围，增强分辨率)
# ==========================================

def generate_stitched_texture(texture_size=(2048, 2048), world_range=20.0):
    # [V32] 扩大 world_range 到 20米，防止截断
    print(f"正在生成全景纹理 ({texture_size[0]}x{texture_size[1]}), 范围: {world_range}m...")
    x = np.linspace(world_range, -world_range, texture_size[0])
    y = np.linspace(world_range, -world_range, texture_size[1])
    grid_x, grid_y = np.meshgrid(x, y, indexing='ij')
    flat_z = np.zeros_like(grid_x)
    world_points = np.stack([grid_x.flatten(), grid_y.flatten(), flat_z.flatten()], axis=1)
    
    cam_configs = [
        {'name': 'FV',  'json': './calib/cam3.json', 'img': './imgs_raw/cam3.png'}, 
        {'name': 'RV',  'json': './calib/cam0.json', 'img': './imgs_raw/cam0.png'}, 
        {'name': 'MVL', 'json': './calib/cam1.json', 'img': './imgs_raw/cam1.png'}, 
        {'name': 'MVR', 'json': './calib/cam2.json', 'img': './imgs_raw/cam2.png'}  
    ]
    cameras, images = [], []
    for cfg in cam_configs:
        if not os.path.exists(cfg['json']): continue
        cameras.append(read_cam(cfg['json'], force_name=cfg['name']))
        images.append(cv2.imread(cfg['img']))
    
    if not cameras: raise FileNotFoundError("No cameras found.")

    # 亮度平衡
    means = [np.mean(cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[:,:,0]) for img in images]
    target = np.mean(means)
    balanced_images = []
    for i, img in enumerate(images):
        gain = np.clip(target / (means[i]+1e-5), 0.8, 1.2)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = cv2.multiply(l, gain)
        l = np.clip(l, 0, 255).astype(np.uint8)
        balanced_images.append(cv2.cvtColor(cv2.merge([l,a,b]), cv2.COLOR_LAB2BGR))
    
    canvas = np.zeros((texture_size[0], texture_size[1], 3), dtype=np.float32)
    total_weights = np.zeros((texture_size[0], texture_size[1]), dtype=np.float32)
    
    for i, cam in enumerate(cameras):
        uv = cam.project_3d_to_2d(world_points)
        map_x = uv[:, 0].reshape(texture_size).astype(np.float32)
        map_y = uv[:, 1].reshape(texture_size).astype(np.float32)
        weight = get_border_distance_weight(map_x, map_y, cam.width, cam.height)
        img = balanced_images[i]
        warped = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        canvas += warped.astype(np.float32) * weight[:, :, np.newaxis]
        total_weights += weight
        
    valid_mask = total_weights > 0.0001
    total_weights[~valid_mask] = 1.0
    final_image = canvas / total_weights[:, :, np.newaxis]
    final_image = np.clip(final_image, 0, 255).astype(np.uint8)
    return final_image, world_range

# ==========================================
# PART 3: OpenGL 交互渲染 (V32 镜像修复)
# ==========================================

rot_x, rot_y = 60, 0 # 初始视角稍微抬高一点
last_x, last_y = 0, 0
is_dragging = False
zoom = -30.0 # 初始拉远一点，看全貌

def mouse_button_callback(window, button, action, mods):
    global is_dragging, last_x, last_y
    if button == glfw.MOUSE_BUTTON_LEFT:
        if action == glfw.PRESS:
            is_dragging = True
            last_x, last_y = glfw.get_cursor_pos(window)
        elif action == glfw.RELEASE:
            is_dragging = False

def cursor_position_callback(window, xpos, ypos):
    global rot_x, rot_y, last_x, last_y
    if is_dragging:
        dx = xpos - last_x
        dy = ypos - last_y
        rot_y += dx * 0.5
        rot_x += dy * 0.5
        last_x, last_y = xpos, ypos

def scroll_callback(window, xoffset, yoffset):
    global zoom
    zoom += yoffset * 2.0 # 加快缩放速度

def create_radial_bowl_mesh(num_rings=64, num_sectors=128, world_range=20.0):
    vertices = []
    tex_coords = []
    indices = []
    
    flat_percent = 0.40 # 稍微减小纯平区域，让过渡更自然
    flat_radius = world_range * flat_percent
    curvature = 0.08    # 减小曲率，让碗更浅，视觉失真更小
    
    for i in range(num_rings + 1):
        r = (i / num_rings) * world_range
        z = 0.0
        if r > flat_radius:
            z = (r - flat_radius)**2 * curvature
            
        for j in range(num_sectors):
            angle = (j / num_sectors) * 2.0 * np.pi
            x = r * np.cos(angle)
            y = r * np.sin(angle)
            
            # OpenGL 坐标: Y-up (x, z, y)
            vertices.extend([x, z, y]) 
            
            # UV 映射
            u = (x + world_range) / (2.0 * world_range)
            v = (world_range - y) / (2.0 * world_range)
            tex_coords.extend([u, v])
            
    for i in range(num_rings):
        for j in range(num_sectors):
            curr_start = i * num_sectors
            next_start = (i + 1) * num_sectors
            curr_sec = j
            next_sec = (j + 1) % num_sectors
            p1 = curr_start + curr_sec
            p2 = curr_start + next_sec
            p3 = next_start + curr_sec
            p4 = next_start + next_sec
            indices.extend([p1, p3, p2])
            indices.extend([p2, p3, p4])
            
    return np.array(vertices, dtype=np.float32), np.array(tex_coords, dtype=np.float32), np.array(indices, dtype=np.uint32)

def main():
    if not glfw.init(): return
    glfw.window_hint(glfw.SAMPLES, 4)
    window = glfw.create_window(1024, 768, "3D AVM V32 - Fixed & Sharp", None, None)
    if not window: glfw.terminate(); return
    glfw.make_context_current(window)
    glfw.set_mouse_button_callback(window, mouse_button_callback)
    glfw.set_cursor_pos_callback(window, cursor_position_callback)
    glfw.set_scroll_callback(window, scroll_callback)
    
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_TEXTURE_2D)
    glEnable(GL_MULTISAMPLE)

    try:
        # [V32] 扩大生成范围
        texture_img, world_range = generate_stitched_texture(texture_size=(2048, 2048), world_range=20.0)
    except Exception as e:
        print(f"Error: {e}")
        glfw.terminate()
        return

    # [V32 关键修复]: 上下翻转图片，解决 OpenGL 镜像问题
    texture_img = cv2.flip(texture_img, 0)
    
    texture_img = cv2.cvtColor(texture_img, cv2.COLOR_BGR2RGB)
    tex_h, tex_w, _ = texture_img.shape
    
    texture_id = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture_id)
    
    # [V32 关键修复]: 禁用 Mipmap，使用 GL_LINEAR 保持近处锐度
    # 也可以尝试 GL_NEAREST 看看像素点，但 GL_LINEAR 通常更平滑
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, tex_w, tex_h, 0, GL_RGB, GL_UNSIGNED_BYTE, texture_img)
    
    vertices, tex_coords, indices = create_radial_bowl_mesh(world_range=world_range)
    
    print("启动 V32 渲染窗口...")
    print("  - 已修复镜像问题 (翻转纹理)")
    print("  - 已提升清晰度 (禁用Mipmap, 2048px)")
    print("  - 已扩大视野 (20m范围)")
    
    while not glfw.window_should_close(window):
        width, height = glfw.get_framebuffer_size(window)
        glViewport(0, 0, width, height)
        glClearColor(0.2, 0.3, 0.3, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, width / (height if height > 0 else 1), 0.1, 100.0)
        
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glTranslatef(0.0, -8.0, zoom) # 相机稍微抬高，以便看到全貌
        glRotatef(rot_x, 1, 0, 0)
        glRotatef(rot_y, 0, 1, 0)
        
        # 绘制车身
        glColor3f(0.1, 0.1, 0.1)
        glPushMatrix()
        glScalef(1.8, 0.6, 3.8) 
        glBegin(GL_QUADS)
        glVertex3f(-0.5, 0.5, -0.5); glVertex3f( 0.5, 0.5, -0.5); glVertex3f( 0.5, 0.5,  0.5); glVertex3f(-0.5, 0.5,  0.5)
        glEnd()
        glPopMatrix()
        
        # 绘制碗
        glColor3f(1.0, 1.0, 1.0)
        glBindTexture(GL_TEXTURE_2D, texture_id)
        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_TEXTURE_COORD_ARRAY)
        glVertexPointer(3, GL_FLOAT, 0, vertices)
        glTexCoordPointer(2, GL_FLOAT, 0, tex_coords)
        glDrawElements(GL_TRIANGLES, len(indices), GL_UNSIGNED_INT, indices)
        glDisableClientState(GL_VERTEX_ARRAY)
        glDisableClientState(GL_TEXTURE_COORD_ARRAY)
        
        glfw.swap_buffers(window)
        glfw.poll_events()

    glfw.terminate()

if __name__ == "__main__":
    main()