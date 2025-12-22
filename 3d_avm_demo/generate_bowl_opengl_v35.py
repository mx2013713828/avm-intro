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
# PART 1: 基础类 (增加背部剔除逻辑)
# ==========================================

class RadialPolyCamProjection:
    def __init__(self, distortion_params):
        self.coefficients = np.asarray(distortion_params)
        self.power = np.array([np.arange(start=1, stop=self.coefficients.size + 1)]).T
    
    def project_3d_to_2d(self, cam_points):
        # cam_points: (N, 3)
        # 计算距离 chi
        chi = np.sqrt(cam_points[:,0]**2 + cam_points[:,1]**2)
        
        # 计算入射角 theta (光轴为Z轴)
        # atan2(z, chi) 算的是与XY平面的夹角，pi/2 - ... 得到与Z轴夹角
        theta = np.pi/2.0 - np.arctan2(cam_points[:,2], chi)
        
        # 多项式计算 rho
        term_powers = np.power(theta[:, np.newaxis], self.power.T)
        rho = np.dot(term_powers, self.coefficients)
        
        # 避免除零
        with np.errstate(divide='ignore', invalid='ignore'):
            scale = rho / chi
            scale[chi == 0] = 0
            
        uv = scale[:, np.newaxis] * cam_points[:, 0:2]
        
        # [V35 新增] 返回 theta 以便进行角度剔除
        return uv, theta

class Camera:
    def __init__(self, json_path, name):
        with open(json_path) as f: config = json.load(f)
        intr = config['intrinsic']
        self.width = intr['width']
        self.height = intr['height']
        self.name = name
        
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
        
        uv_norm = np.stack([u_norm, v_norm], axis=1)
        
        # [V35 核心修复]: 背部投影剔除 (Back-Projection Culling)
        # 任何入射角超过 100 度 (1.74弧度) 的点，都认为是“背后”的点
        # 或者是 Z < 0 的点 (虽然鱼眼可以看一点背后，但太远的不行)
        
        # 这里的 cam_pts[:, 2] 是 Z (光轴方向)
        # 如果 Z < -0.1 (在相机背面)，且 chi 很大，就是错误的
        
        # 我们使用一个掩码：只有 theta 在合理范围内 (例如 0 ~ 110度) 才有效
        valid_mask = (theta >= 0) & (theta < np.deg2rad(110.0))
        
        return uv_norm, valid_mask

# ==========================================
# PART 2: 矩阵数学
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
    # 构建 View 矩阵 (逆变换)
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
# PART 3: Shader (稍微调整背景色)
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

uniform int debugMode;

void main()
{
    if (debugMode == 1) {
        vec3 wColor = vec3(Weights.x, Weights.y, Weights.z) + vec3(Weights.w, Weights.w, 0.0);
        FragColor = vec4(wColor, 1.0);
        return;
    }

    vec4 colFV = texture(texFV, UV_FV);
    vec4 colRV = texture(texRV, UV_RV);
    vec4 colMVL = texture(texMVL, UV_MVL);
    vec4 colMVR = texture(texMVR, UV_MVR);
    
    float wTotal = Weights.x + Weights.y + Weights.z + Weights.w;
    
    // 如果没有相机覆盖这里，显示黑色或网格
    if (wTotal < 0.001) {
        FragColor = vec4(0.05, 0.05, 0.05, 1.0); // 几乎全黑
        return;
    }
    
    vec4 finalColor = colFV * Weights.x + colRV * Weights.y + colMVL * Weights.z + colMVR * Weights.w;
    
    // 简单的 Gamma 校正，让画面亮一点
    vec3 color = finalColor.rgb / wTotal;
    color = pow(color, vec3(1.0/1.1)); 
    
    FragColor = vec4(color, 1.0);
}
"""

# ==========================================
# PART 4: LUT 生成 (增加翻转修正)
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
            
            # [V35 修复]: 镜像修正
            # 如果之前的画面左右是反的，我们在这里翻转 X 坐标
            # 通常 CV 坐标系和 GL 坐标系在 X 轴上定义不同
            x = -x 
            
            grid_pts.append([x, y, z]) 

    grid_pts = np.array(grid_pts, dtype=np.float32)
    
    # 批量计算 LUT
    uv_maps = []
    valid_masks = []
    
    for cam in cameras:
        # 传入所有世界点
        uv, mask = cam.get_uv_for_world_points(grid_pts)
        uv_maps.append(uv)
        valid_masks.append(mask)
        
    weights = np.zeros((len(grid_pts), 4), dtype=np.float32)
    for k in range(4):
        u = uv_maps[k][:, 0]
        v = uv_maps[k][:, 1]
        
        # 只有在 UV 范围内，且 valid_mask 为 True (不在相机背后) 的点才有效
        is_in_image = (u > 0) & (u < 1) & (v > 0) & (v < 1)
        valid = is_in_image & valid_masks[k]
        
        # 边缘融合权重
        dist_edge = np.minimum(np.minimum(u, 1-u), np.minimum(v, 1-v))
        w = np.maximum(0, dist_edge)
        w = np.power(w, 0.5)
        
        weights[:, k] = w * valid.astype(np.float32)

    # GL Y-up: World Z -> GL Y
    # World (x, y, z_height) -> GL (x, z_height, y) 
    # 注意: grid_pts[1] 是 World Y
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
debug_mode = 0
wireframe_mode = False

def main():
    if not glfw.init(): return
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    
    window = glfw.create_window(1024, 768, "V35: Back-Culling Fixed", None, None)
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

    print("Loading Data...")
    cam_files = ['./calib/cam3.json', './calib/cam0.json', './calib/cam1.json', './calib/cam2.json']
    img_files = ['./imgs_raw/cam3.png', './imgs_raw/cam0.png', './imgs_raw/cam1.png', './imgs_raw/cam2.png']
    
    cameras = []
    for i, (cf, imf) in enumerate(zip(cam_files, img_files)):
        if not os.path.exists(cf): continue
        cameras.append(Camera(cf, f"Cam{i}"))
        
        img = cv2.imread(imf)
        img = cv2.flip(img, 0) # Flip V for GL
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        tid = glGenTextures(1)
        glActiveTexture(GL_TEXTURE0 + i)
        glBindTexture(GL_TEXTURE_2D, tid)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, img.shape[1], img.shape[0], 0, GL_RGB, GL_UNSIGNED_BYTE, img)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER)
        
        loc = glGetUniformLocation(shader, ["texFV", "texRV", "texMVL", "texMVR"][i])
        glUniform1i(loc, i)

    print("Generating Mesh (with Back-Face Culling)...")
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
    debug_loc = glGetUniformLocation(shader, "debugMode")

    def key_cb(w, key, scancode, action, mods):
        global debug_mode, wireframe_mode
        if action == glfw.PRESS:
            if key == glfw.KEY_C: debug_mode = 1 - debug_mode
            if key == glfw.KEY_W:
                wireframe_mode = not wireframe_mode
                glPolygonMode(GL_FRONT_AND_BACK, GL_LINE if wireframe_mode else GL_FILL)

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

    print("Rendering V35... Clean View.")
    
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
        
        # 4. 调整视角矩阵逻辑以匹配 "X=Front" 
        # (因为我们在 mesh 生成时翻转了 X，这里可能需要相应调整)
        eye_pos = np.array([cam_x, cam_y, cam_z], dtype=np.float32)

        aspect = w / h if h > 0 else 1.0
        proj_mat = create_perspective_matrix(60, aspect, 0.1, 100.0)
        
        # 简单的 LookAt (不依赖外部库)
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
        glUniform1i(debug_loc, debug_mode)

        glBindVertexArray(vao)
        glDrawElements(GL_TRIANGLES, len(i_data), GL_UNSIGNED_INT, None)
        
        glfw.swap_buffers(window)
        glfw.poll_events()

    glfw.terminate()

if __name__ == "__main__":
    main()