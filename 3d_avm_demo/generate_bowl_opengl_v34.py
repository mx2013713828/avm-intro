#!/usr/bin/env python3
import json
import numpy as np
import cv2
import os
import glfw
import time
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
from scipy.spatial.transform import Rotation as SciRot

# ==========================================
# PART 1: 数学工具库 (增强健壮性)
# ==========================================

def create_perspective_matrix(fov_deg, aspect, near, far):
    # 标准 OpenGL 透视投影矩阵
    # 避免 aspect 为 0
    if aspect < 0.0001: aspect = 1.0
    
    f = 1.0 / np.tan(np.radians(fov_deg) / 2)
    mat = np.zeros((4, 4), dtype=np.float32)
    
    # Row-Major 布局 (配合 glUniformMatrix4fv(..., GL_TRUE, ...))
    mat[0, 0] = f / aspect
    mat[1, 1] = f
    mat[2, 2] = (far + near) / (near - far)
    mat[2, 3] = (2 * far * near) / (near - far)
    mat[3, 2] = -1.0
    mat[3, 3] = 0.0
    return mat

def look_at(eye, center, up):
    # 标准 LookAt 矩阵实现
    # eye: 相机位置
    # center: 看向的点
    # up: 上方向
    z_axis = eye - center
    z_axis = z_axis / (np.linalg.norm(z_axis) + 1e-6)
    
    x_axis = np.cross(up, z_axis)
    x_axis = x_axis / (np.linalg.norm(x_axis) + 1e-6)
    
    y_axis = np.cross(z_axis, x_axis)
    
    # 构建视图矩阵 (Rotation * Translation)
    view = np.eye(4, dtype=np.float32)
    view[0, 0:3] = x_axis
    view[1, 0:3] = y_axis
    view[2, 0:3] = z_axis
    view[0, 3] = -np.dot(x_axis, eye)
    view[1, 3] = -np.dot(y_axis, eye)
    view[2, 3] = -np.dot(z_axis, eye)
    
    return view

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
        return uv

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
        ones = np.ones((world_points.shape[0], 1))
        pts_homo = np.hstack([world_points, ones])
        cam_pts = pts_homo @ self.inv_pose.T
        uv_distorted = self.lens.project_3d_to_2d(cam_pts[:, :3])
        uv_pixel = (uv_distorted * self.aspect_ratio) + self.principle_point
        u_norm = uv_pixel[:, 0] / self.width
        v_norm = uv_pixel[:, 1] / self.height
        return np.stack([u_norm, v_norm], axis=1)

# ==========================================
# PART 2: Shader (增加调试模式)
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
out vec3 DebugPos; // 传递世界坐标用于调试

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

uniform int debugMode; // 0=正常, 1=显示UV分布

void main()
{
    // 调试模式：显示UV坐标和权重，不显示纹理
    if (debugMode == 1) {
        // 将权重可视化为颜色：前=红, 后=绿, 左=蓝, 右=黄
        vec3 wColor = vec3(Weights.x, Weights.y, Weights.z) + vec3(Weights.w, Weights.w, 0.0);
        FragColor = vec4(wColor, 1.0);
        return;
    }

    vec4 colFV = texture(texFV, UV_FV);
    vec4 colRV = texture(texRV, UV_RV);
    vec4 colMVL = texture(texMVL, UV_MVL);
    vec4 colMVR = texture(texMVR, UV_MVR);
    
    float wTotal = Weights.x + Weights.y + Weights.z + Weights.w;
    if (wTotal < 0.001) {
        // 无权重区域显示灰色网格线
        float grid = step(0.95, fract(DebugPos.x)) + step(0.95, fract(DebugPos.z));
        FragColor = vec4(vec3(0.2) + grid*0.3, 1.0);
        return;
    }
    
    vec4 finalColor = colFV * Weights.x + colRV * Weights.y + colMVL * Weights.z + colMVR * Weights.w;
    FragColor = finalColor / wTotal;
}
"""

# ==========================================
# PART 3: 数据生成 (带诊断)
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
            grid_pts.append([x, y, z]) 

    grid_pts = np.array(grid_pts, dtype=np.float32)
    
    # [DEBUG] 打印网格统计信息
    print(f"[DEBUG] Mesh Generated: {len(grid_pts)} vertices")
    print(f"[DEBUG]   X Range: {grid_pts[:,0].min():.2f} to {grid_pts[:,0].max():.2f}")
    print(f"[DEBUG]   Z Range: {grid_pts[:,2].min():.2f} to {grid_pts[:,2].max():.2f} (Height)")
    print(f"[DEBUG]   Y Range: {grid_pts[:,1].min():.2f} to {grid_pts[:,1].max():.2f}")
    
    uv_maps = []
    for cam in cameras:
        uv = cam.get_uv_for_world_points(grid_pts)
        uv_maps.append(uv)
        
    weights = np.zeros((len(grid_pts), 4), dtype=np.float32)
    for k in range(4):
        u = uv_maps[k][:, 0]
        v = uv_maps[k][:, 1]
        valid = (u > 0) & (u < 1) & (v > 0) & (v < 1)
        dist_edge = np.minimum(np.minimum(u, 1-u), np.minimum(v, 1-v))
        w = np.maximum(0, dist_edge)
        w = np.power(w, 0.5)
        weights[:, k] = w * valid.astype(np.float32)

    # GL Y-up: World Z -> GL Y
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
# PART 4: 主程序
# ==========================================

# 交互状态
cam_dist = 20.0
cam_pitch = 45.0 # 俯仰角
cam_yaw = 0.0    # 旋转角
last_x, last_y = 0, 0
is_dragging = False
debug_mode = 0  # 0: Texture, 1: Color
wireframe_mode = False

def main():
    if not glfw.init(): return
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    
    window = glfw.create_window(1024, 768, "V34: Debug Mode (Press W:Wireframe, C:Color)", None, None)
    if not window: glfw.terminate(); return
    glfw.make_context_current(window)
    
    glEnable(GL_DEPTH_TEST)
    
    # 编译 Shader
    try:
        shader = compileProgram(
            compileShader(VERTEX_SHADER, GL_VERTEX_SHADER),
            compileShader(FRAGMENT_SHADER, GL_FRAGMENT_SHADER)
        )
    except Exception as e:
        print(f"Shader Error: {e}")
        return
    glUseProgram(shader)

    # 加载数据
    print("Loading Data...")
    cam_files = ['./calib/cam3.json', './calib/cam0.json', './calib/cam1.json', './calib/cam2.json']
    img_files = ['./imgs_raw/cam3.png', './imgs_raw/cam0.png', './imgs_raw/cam1.png', './imgs_raw/cam2.png']
    
    cameras = []
    for i, (cf, imf) in enumerate(zip(cam_files, img_files)):
        if not os.path.exists(cf): continue
        cameras.append(Camera(cf, f"Cam{i}"))
        
        img = cv2.imread(imf)
        img = cv2.flip(img, 0) # Flip V
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

    # Uniform Locations
    model_loc = glGetUniformLocation(shader, "model")
    view_loc = glGetUniformLocation(shader, "view")
    proj_loc = glGetUniformLocation(shader, "projection")
    debug_loc = glGetUniformLocation(shader, "debugMode")

    # Callbacks
    def key_cb(w, key, scancode, action, mods):
        global debug_mode, wireframe_mode
        if action == glfw.PRESS:
            if key == glfw.KEY_C:
                debug_mode = 1 - debug_mode
                print(f"[DEBUG] Debug Color Mode: {'ON' if debug_mode else 'OFF'}")
            if key == glfw.KEY_W:
                wireframe_mode = not wireframe_mode
                glPolygonMode(GL_FRONT_AND_BACK, GL_LINE if wireframe_mode else GL_FILL)
                print(f"[DEBUG] Wireframe Mode: {'ON' if wireframe_mode else 'OFF'}")

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
            cam_pitch = max(10, min(89, cam_pitch)) # 限制俯仰角，防止反转
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
    print(" [Left Click + Drag]: Rotate Camera")
    print(" [Scroll]: Zoom In/Out")
    print(" [W]: Toggle Wireframe (Check Mesh Shape)")
    print(" [C]: Toggle Color Debug (Check Weights)")
    print("=======================================")

    last_print = 0
    while not glfw.window_should_close(window):
        t = glfw.get_time()
        
        w, h = glfw.get_framebuffer_size(window)
        glViewport(0, 0, w, h)
        glClearColor(0.1, 0.1, 0.1, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        # 1. 计算相机位置 (球坐标 -> 笛卡尔坐标)
        # 始终看向 (0,0,0)
        rad_pitch = np.radians(cam_pitch)
        rad_yaw = np.radians(cam_yaw)
        
        # OpenGL Y-Up
        cam_y = cam_dist * np.sin(rad_pitch)
        cam_h = cam_dist * np.cos(rad_pitch)
        cam_x = cam_h * np.sin(rad_yaw)
        cam_z = cam_h * np.cos(rad_yaw)
        
        eye_pos = np.array([cam_x, cam_y, cam_z], dtype=np.float32)
        center_pos = np.array([0, 0, 0], dtype=np.float32)
        up_vec = np.array([0, 1, 0], dtype=np.float32)
        
        # 2. 构建矩阵
        aspect = w / h if h > 0 else 1.0
        proj_mat = create_perspective_matrix(60, aspect, 0.1, 100.0)
        view_mat = look_at(eye_pos, center_pos, up_vec)
        model_mat = np.eye(4, dtype=np.float32)
        
        # 3. 诊断输出 (每2秒一次)
        if t - last_print > 2.0:
            print(f"[DEBUG] Cam Pos: ({eye_pos[0]:.1f}, {eye_pos[1]:.1f}, {eye_pos[2]:.1f}) | Pitch: {cam_pitch:.1f}")
            last_print = t

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