#!/usr/bin/env python3
import json
import numpy as np
import cv2
import os
import glfw
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
from scipy.spatial.transform import Rotation as SciRot

# ==========================================
# PART 1: 投影与相机类
# ==========================================

class RadialPolyCamProjection:
    def __init__(self, distortion_params):
        self.coefficients = np.asarray(distortion_params)
        self.power = np.array([np.arange(start=1, stop=self.coefficients.size + 1)]).T
    
    def project_3d_to_2d(self, cam_points):
        chi = np.sqrt(cam_points[:,0]**2 + cam_points[:,1]**2)
        theta = np.pi/2.0 - np.arctan2(cam_points[:,2], chi)
        term_powers = np.power(theta[:, np.newaxis], self.power.T)
        rho = np.dot(term_powers, self.coefficients)
        with np.errstate(divide='ignore', invalid='ignore'):
            scale = rho / chi
            scale[chi == 0] = 0
        uv = scale[:, np.newaxis] * cam_points[:, 0:2]
        return uv, theta

class Camera:
    def __init__(self, json_path, name, cam_id):
        with open(json_path) as f: config = json.load(f)
        intr = config['intrinsic']
        self.width = intr['width']
        self.height = intr['height']
        self.name = name
        self.id = cam_id # 0=FV, 1=RV, 2=MVL, 3=MVR
        
        self.lens = RadialPolyCamProjection([intr['k1'], intr['k2'], intr['k3'], intr['k4']])
        self.principle_point = np.array([intr['cx_offset'], intr['cy_offset']]) + np.array([self.width, self.height])/2.0 - 0.5
        self.aspect_ratio = np.array([1, intr['aspect_ratio']])
        
        rot = SciRot.from_quat(config['extrinsic']['quaternion']).as_matrix()
        pos = np.array(config['extrinsic']['translation'])
        pose = np.eye(4); pose[:3,:3]=rot; pose[:3,3]=pos
        self.inv_pose = np.linalg.inv(pose)

    def get_uv_for_world_points(self, world_points):
        # 1. World -> Camera
        ones = np.ones((world_points.shape[0], 1))
        pts_homo = np.hstack([world_points, ones])
        cam_pts = pts_homo @ self.inv_pose.T
        
        # 2. Camera -> Lens -> Screen
        uv_distorted, theta = self.lens.project_3d_to_2d(cam_pts[:, :3])
        uv_pixel = (uv_distorted * self.aspect_ratio) + self.principle_point
        
        # 3. Normalize
        u_norm = uv_pixel[:, 0] / self.width
        v_norm = uv_pixel[:, 1] / self.height
        
        # [V36 核心修复]：扇区强制剔除 (Sector Culling)
        # 我们不仅剔除背后点，还要确保前视相机只投射到碗的“前方”。
        # Woodscape 坐标系 (假设): X=前, Y=左
        # 计算世界坐标点的方位角
        # world_points 是 [x, y, z]
        pts_angle = np.arctan2(world_points[:, 1], world_points[:, 0]) # (-pi, pi)
        
        # 定义每个相机的“主方向”角度 (弧度)
        # 这是一个猜测，如果结果不对，我们需要调整这里的偏移量
        # 假设：FV=0度(X轴), MVL=90度(Y轴), RV=180度(-X轴), MVR=-90度(-Y轴)
        cam_centers = {
            0: 0.0,           # FV
            1: np.pi,         # RV
            2: np.pi/2.0,     # MVL
            3: -np.pi/2.0     # MVR
        }
        target_angle = cam_centers[self.id]
        
        # 计算角度差 (考虑周期性)
        angle_diff = np.abs(np.arctan2(np.sin(pts_angle - target_angle), np.cos(pts_angle - target_angle)))
        
        # 允许的 FOV 半径：比如只允许 +/- 100 度以内的点
        # 超过这个范围的点，即使数学上能投影，也强制剔除，防止乱飞
        sector_mask = angle_diff < np.deg2rad(100.0)
        
        return np.stack([u_norm, v_norm], axis=1), sector_mask

# ==========================================
# PART 2: Shader (增加单相机调试)
# ==========================================

VERTEX_SHADER = """
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec2 aUV_FV;
layout (location = 2) in vec2 aUV_RV;
layout (location = 3) in vec2 aUV_MVL;
layout (location = 4) in vec2 aUV_MVR;
layout (location = 5) in vec4 aWeights;

out vec2 UV_FV;
out vec2 UV_RV;
out vec2 UV_MVL;
out vec2 UV_MVR;
out vec4 Weights;
out vec3 DebugPos;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main()
{
    gl_Position = projection * view * model * vec4(aPos, 1.0);
    UV_FV = aUV_FV;
    UV_RV = aUV_RV;
    UV_MVL = aUV_MVL;
    UV_MVR = aUV_MVR;
    Weights = aWeights;
    DebugPos = aPos;
}
"""

FRAGMENT_SHADER = """
#version 330 core
out vec4 FragColor;

in vec2 UV_FV;
in vec2 UV_RV;
in vec2 UV_MVL;
in vec2 UV_MVR;
in vec4 Weights;
in vec3 DebugPos;

uniform sampler2D texFV;
uniform sampler2D texRV;
uniform sampler2D texMVL;
uniform sampler2D texMVR;

// 0=All, 1=FV, 2=RV, 3=MVL, 4=MVR
uniform int activeCam; 

void main()
{
    vec4 colFV = texture(texFV, UV_FV);
    vec4 colRV = texture(texRV, UV_RV);
    vec4 colMVL = texture(texMVL, UV_MVL);
    vec4 colMVR = texture(texMVR, UV_MVR);
    
    vec4 finalColor = vec4(0.0);
    float wTotal = 0.0;
    
    // 单相机调试逻辑
    if (activeCam == 1) { // Only FV
        if (Weights.x > 0.001) FragColor = colFV; 
        else FragColor = vec4(0.1, 0.0, 0.0, 1.0); // 红色背景表示无覆盖
        return;
    }
    if (activeCam == 2) { // Only RV
        if (Weights.y > 0.001) FragColor = colRV;
        else FragColor = vec4(0.0, 0.1, 0.0, 1.0);
        return;
    }
    if (activeCam == 3) { // Only MVL
        if (Weights.z > 0.001) FragColor = colMVL;
        else FragColor = vec4(0.0, 0.0, 0.1, 1.0);
        return;
    }
    if (activeCam == 4) { // Only MVR
        if (Weights.w > 0.001) FragColor = colMVR;
        else FragColor = vec4(0.1, 0.1, 0.0, 1.0);
        return;
    }

    // 正常融合模式
    wTotal = Weights.x + Weights.y + Weights.z + Weights.w;
    if (wTotal < 0.001) {
        float grid = step(0.95, fract(DebugPos.x*0.5)) + step(0.95, fract(DebugPos.z*0.5));
        FragColor = vec4(vec3(0.1) + grid*0.2, 1.0);
        return;
    }
    
    finalColor = colFV * Weights.x + colRV * Weights.y + colMVL * Weights.z + colMVR * Weights.w;
    FragColor = finalColor / wTotal;
}
"""

# ==========================================
# PART 3: 矩阵数学
# ==========================================

def create_perspective_matrix(fov_deg, aspect, near, far):
    if aspect < 0.0001: aspect = 1.0
    f = 1.0 / np.tan(np.radians(fov_deg) / 2)
    mat = np.zeros((4, 4), dtype=np.float32)
    mat[0, 0] = f / aspect
    mat[1, 1] = f
    mat[2, 2] = (far + near) / (near - far)
    mat[2, 3] = (2 * far * near) / (near - far)
    mat[3, 2] = -1.0
    mat[3, 3] = 0.0
    return mat

def create_view_matrix(cam_pos, rot_x, rot_y):
    t_inv = np.eye(4, dtype=np.float32)
    t_inv[3, 0] = -cam_pos[0]
    t_inv[3, 1] = -cam_pos[1]
    t_inv[3, 2] = -cam_pos[2]
    
    ry_inv = np.eye(4, dtype=np.float32)
    angle_y = np.radians(-rot_y)
    c, s = np.cos(angle_y), np.sin(angle_y)
    ry_inv[0, 0] = c; ry_inv[0, 2] = -s
    ry_inv[2, 0] = s; ry_inv[2, 2] = c
    
    rx_inv = np.eye(4, dtype=np.float32)
    angle_x = np.radians(-rot_x)
    c, s = np.cos(angle_x), np.sin(angle_x)
    rx_inv[1, 1] = c; rx_inv[1, 2] = s
    rx_inv[2, 1] = -s; rx_inv[2, 2] = c
    return rx_inv @ ry_inv @ t_inv

# ==========================================
# PART 4: LUT 生成 (世界坐标系旋转)
# ==========================================

def create_bowl_data_lut(cameras, world_range=20.0, rings=80, sectors=128):
    flat_percent = 0.40
    flat_radius = world_range * flat_percent
    curvature = 0.08
    
    grid_pts = []
    for i in range(rings + 1):
        r = (i / rings) * world_range
        z = 0.0
        if r > flat_radius:
            z = (r - flat_radius)**2 * curvature
        for j in range(sectors):
            angle = (j / sectors) * 2.0 * np.pi
            x = r * np.cos(angle)
            y = r * np.sin(angle)
            
            # [V36 尝试]: 这里可能需要旋转世界坐标系以匹配相机定义
            # Woodscape 可能是前=X, 左=Y。
            # 如果画面错位，我们可以尝试在这里交换 x, y 或取反
            # 目前保持默认，依赖 Sector Culling 来清理
            
            grid_pts.append([x, y, z]) 

    grid_pts = np.array(grid_pts, dtype=np.float32)
    
    uv_maps = []
    valid_masks = []
    
    for cam in cameras:
        uv, mask = cam.get_uv_for_world_points(grid_pts)
        uv_maps.append(uv)
        valid_masks.append(mask)
        
    weights = np.zeros((len(grid_pts), 4), dtype=np.float32)
    for k in range(4):
        u = uv_maps[k][:, 0]
        v = uv_maps[k][:, 1]
        
        # [V36] 权重计算：结合 UV 范围 + Sector Mask
        is_in_image = (u > 0) & (u < 1) & (v > 0) & (v < 1)
        valid = is_in_image & valid_masks[k]
        
        dist_edge = np.minimum(np.minimum(u, 1-u), np.minimum(v, 1-v))
        w = np.maximum(0, dist_edge)
        w = np.power(w, 0.5)
        
        weights[:, k] = w * valid.astype(np.float32)

    # GL Y-up: World (x, y, z_height) -> GL (x, z_height, y)
    gl_pos = np.stack([grid_pts[:, 0], grid_pts[:, 2], grid_pts[:, 1]], axis=1)

    indices = []
    for i in range(rings):
        for j in range(sectors):
            curr = i * sectors + j
            next_row = (i+1) * sectors + j
            curr_next = i * sectors + (j+1)%sectors
            next_row_next = (i+1) * sectors + (j+1)%sectors
            indices.extend([curr, next_row, curr_next])
            indices.extend([curr_next, next_row, next_row_next])

    vertex_data = np.hstack([
        gl_pos, uv_maps[0], uv_maps[1], uv_maps[2], uv_maps[3], weights
    ]).flatten().astype(np.float32)
    
    return vertex_data, np.array(indices, dtype=np.uint32)

# ==========================================
# PART 5: 主程序
# ==========================================

cam_dist = 25.0
cam_pitch = 45.0
cam_yaw = 0.0
last_x, last_y = 0, 0
is_dragging = False
active_cam = 0 # 0=All

def main():
    if not glfw.init(): return
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    
    window = glfw.create_window(1024, 768, "V36: Press 1/2/3/4 to check cameras!", None, None)
    if not window: glfw.terminate(); return
    glfw.make_context_current(window)
    glEnable(GL_DEPTH_TEST)
    
    try:
        shader = compileProgram(
            compileShader(VERTEX_SHADER, GL_VERTEX_SHADER),
            compileShader(FRAGMENT_SHADER, GL_FRAGMENT_SHADER)
        )
    except Exception as e:
        print(f"Shader Error: {e}")
        return
    glUseProgram(shader)

    cam_files = ['./calib/cam3.json', './calib/cam0.json', './calib/cam1.json', './calib/cam2.json'] # FV, RV, MVL, MVR
    img_files = ['./imgs_raw/cam3.png', './imgs_raw/cam0.png', './imgs_raw/cam1.png', './imgs_raw/cam2.png']
    
    cameras = []
    for i, (cf, imf) in enumerate(zip(cam_files, img_files)):
        if not os.path.exists(cf): continue
        cameras.append(Camera(cf, f"Cam{i}", i)) # Pass ID for sector culling
        
        img = cv2.imread(imf)
        img = cv2.flip(img, 0)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        tid = glGenTextures(1)
        glActiveTexture(GL_TEXTURE0 + i)
        glBindTexture(GL_TEXTURE_2D, tid)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, img.shape[1], img.shape[0], 0, GL_RGB, GL_UNSIGNED_BYTE, img)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER)
        glUniform1i(glGetUniformLocation(shader, ["texFV", "texRV", "texMVL", "texMVR"][i]), i)

    print("Generating Mesh...")
    v_data, i_data = create_bowl_data_lut(cameras, world_range=20.0)
    
    vao = glGenVertexArrays(1)
    glBindVertexArray(vao)
    vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, v_data.nbytes, v_data, GL_STATIC_DRAW)
    ebo = glGenBuffers(1)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, i_data.nbytes, i_data, GL_STATIC_DRAW)
    
    stride = 15 * 4
    for i in range(6):
        glVertexAttribPointer(i, 3 if i==0 else (4 if i==5 else 2), GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p([0,12,20,28,36,44][i]))
        glEnableVertexAttribArray(i)

    model_loc = glGetUniformLocation(shader, "model")
    view_loc = glGetUniformLocation(shader, "view")
    proj_loc = glGetUniformLocation(shader, "projection")
    active_cam_loc = glGetUniformLocation(shader, "activeCam")

    def key_cb(w, key, scancode, action, mods):
        global active_cam
        if action == glfw.PRESS:
            if key == glfw.KEY_0: active_cam = 0
            if key == glfw.KEY_1: active_cam = 1
            if key == glfw.KEY_2: active_cam = 2
            if key == glfw.KEY_3: active_cam = 3
            if key == glfw.KEY_4: active_cam = 4
            print(f"[DEBUG] Active Cam: {active_cam} (0=All, 1=FV, 2=RV, 3=MVL, 4=MVR)")

    def mouse_cb(w, b, a, m):
        global is_dragging, last_x, last_y
        if b == glfw.MOUSE_BUTTON_LEFT:
            if a == glfw.PRESS:
                is_dragging = True
                last_x, last_y = glfw.get_cursor_pos(w)
            elif a == glfw.RELEASE:
                is_dragging = False
    
    def cursor_cb(w, x, y):
        global cam_yaw, cam_pitch, last_x, last_y
        if is_dragging:
            cam_yaw += (x - last_x) * 0.5
            cam_pitch += (y - last_y) * 0.5
            cam_pitch = max(10, min(89, cam_pitch))
            last_x, last_y = x, y
            
    def scroll_cb(w, x, y):
        global cam_dist
        cam_dist -= y * 1.0
        cam_dist = max(2.0, min(60.0, cam_dist))

    glfw.set_key_callback(window, key_cb)
    glfw.set_mouse_button_callback(window, mouse_cb)
    glfw.set_cursor_pos_callback(window, cursor_cb)
    glfw.set_scroll_callback(window, scroll_cb)

    print("=======================================")
    print(" CONTROLS:")
    print(" [1] Front Camera Only")
    print(" [2] Rear Camera Only")
    print(" [3] Left Camera Only")
    print(" [4] Right Camera Only")
    print(" [0] Show All (Fusion)")
    print("=======================================")

    while not glfw.window_should_close(window):
        w, h = glfw.get_framebuffer_size(window)
        glViewport(0, 0, w, h)
        glClearColor(0.1, 0.1, 0.1, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        rad_pitch = np.radians(cam_pitch)
        rad_yaw = np.radians(cam_yaw)
        cam_y = cam_dist * np.sin(rad_pitch)
        cam_h = cam_dist * np.cos(rad_pitch)
        cam_x = cam_h * np.sin(rad_yaw)
        cam_z = cam_h * np.cos(rad_yaw)
        eye_pos = np.array([cam_x, cam_y, cam_z], dtype=np.float32)

        aspect = w / h if h > 0 else 1.0
        proj_mat = create_perspective_matrix(60, aspect, 0.1, 100.0)
        view_mat = create_view_matrix(eye_pos, cam_pitch, cam_yaw) # Simplied view creation
        # Re-using previous robust LookAt function logic inside main loop for clarity
        center = np.zeros(3)
        up = np.array([0,1,0])
        z_axis = eye_pos - center; z_axis /= np.linalg.norm(z_axis)
        x_axis = np.cross(up, z_axis); x_axis /= np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)
        view_mat = np.eye(4, dtype=np.float32)
        view_mat[0, :3] = x_axis; view_mat[1, :3] = y_axis; view_mat[2, :3] = z_axis
        view_mat[0, 3] = -np.dot(x_axis, eye_pos)
        view_mat[1, 3] = -np.dot(y_axis, eye_pos)
        view_mat[2, 3] = -np.dot(z_axis, eye_pos)

        model_mat = np.eye(4, dtype=np.float32)

        glUniformMatrix4fv(proj_loc, 1, GL_TRUE, proj_mat)
        glUniformMatrix4fv(view_loc, 1, GL_TRUE, view_mat)
        glUniformMatrix4fv(model_loc, 1, GL_TRUE, model_mat)
        glUniform1i(active_cam_loc, active_cam)

        glBindVertexArray(vao)
        glDrawElements(GL_TRIANGLES, len(i_data), GL_UNSIGNED_INT, None)
        
        glfw.swap_buffers(window)
        glfw.poll_events()

    glfw.terminate()

if __name__ == "__main__":
    main()