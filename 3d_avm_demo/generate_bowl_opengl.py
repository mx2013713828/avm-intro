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
# PART 1: 基础类 (数学工具)
# ==========================================
# ... (RadialPolyCamProjection, Camera 等类保持不变，直接复用 V32 的定义)
# 为节省篇幅，这里假设你已经引入了这些类，或者我可以为你完整写出。
# 鉴于我们需要精准控制，这里重新定义精简版 Camera 类

class RadialPolyCamProjection:
    def __init__(self, distortion_params):
        self.coefficients = np.asarray(distortion_params)
        self.power = np.array([np.arange(start=1, stop=self.coefficients.size + 1)]).T
    
    def project_3d_to_2d(self, cam_points):
        # 简化版投影，不做裁剪，只负责算 UV
        chi = np.sqrt(cam_points[:,0]**2 + cam_points[:,1]**2)
        theta = np.pi/2.0 - np.arctan2(cam_points[:,2], chi)
        rho = np.dot(np.power(theta[:, None], self.power.T), self.coefficients)
        
        # 避免除以0
        with np.errstate(divide='ignore', invalid='ignore'):
            scale = rho / chi
            scale[chi == 0] = 0
            
        uv = scale[:, None] * cam_points[:, 0:2]
        return uv

class Camera:
    def __init__(self, json_path, name):
        with open(json_path) as f: config = json.load(f)
        intr = config['intrinsic']
        self.width = intr['width']
        self.height = intr['height']
        self.name = name
        
        # 内参
        self.lens = RadialPolyCamProjection([intr['k1'], intr['k2'], intr['k3'], intr['k4']])
        self.principle_point = np.array([intr['cx_offset'], intr['cy_offset']]) + np.array(self.width, self.height)/2.0 - 0.5
        self.aspect_ratio = np.array([1, intr['aspect_ratio']])
        
        # 外参
        rot = SciRot.from_quat(config['extrinsic']['quaternion']).as_matrix()
        pos = np.array(config['extrinsic']['translation'])
        pose = np.eye(4); pose[:3,:3]=rot; pose[:3,3]=pos
        self.inv_pose = np.linalg.inv(pose)

    def get_uv_for_world_points(self, world_points):
        # 1. World -> Camera
        # world_points: (N, 3)
        ones = np.ones((world_points.shape[0], 1))
        pts_homo = np.hstack([world_points, ones])
        cam_pts = pts_homo @ self.inv_pose.T
        
        # 2. Camera -> Lens -> Screen
        uv_distorted = self.lens.project_3d_to_2d(cam_pts[:, :3])
        uv_pixel = (uv_distorted * self.aspect_ratio) + self.principle_point
        
        # 3. Normalize to [0, 1] for OpenGL
        # 注意: OpenGL 纹理坐标原点在左下，OpenCV 在左上
        # 我们需要在 Shader 里或者这里翻转 Y。这里我们在 Python 端归一化，Shader 端采样。
        u_norm = uv_pixel[:, 0] / self.width
        v_norm = uv_pixel[:, 1] / self.height
        
        return np.stack([u_norm, v_norm], axis=1)

# ==========================================
# PART 2: Shader 代码
# ==========================================

VERTEX_SHADER = """
#version 330 core
layout (location = 0) in vec3 aPos;       // 顶点坐标
layout (location = 1) in vec2 aUV_FV;     // 前视 UV
layout (location = 2) in vec2 aUV_RV;     // 后视 UV
layout (location = 3) in vec2 aUV_MVL;    // 左视 UV
layout (location = 4) in vec2 aUV_MVR;    // 右视 UV
layout (location = 5) in vec4 aWeights;   // 4个相机的混合权重

out vec2 UV_FV;
out vec2 UV_RV;
out vec2 UV_MVL;
out vec2 UV_MVR;
out vec4 Weights;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main()
{
    gl_Position = projection * view * model * vec4(aPos, 1.0);
    
    // 直接把计算好的 UV 透传给 Fragment Shader
    UV_FV = aUV_FV;
    UV_RV = aUV_RV;
    UV_MVL = aUV_MVL;
    UV_MVR = aUV_MVR;
    Weights = aWeights;
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

// 4个纹理采样器
uniform sampler2D texFV;
uniform sampler2D texRV;
uniform sampler2D texMVL;
uniform sampler2D texMVR;

void main()
{
    // 在显卡上直接读取 4 张高清原图
    // 注意: 原图 Y 轴需要翻转? 取决于上传时的设置。通常 OpenCV 图片上传后是倒的。
    // 这里假设我们上传前已经 Flip 了，或者 UV 已经处理好了。
    
    vec4 colFV = texture(texFV, UV_FV);
    vec4 colRV = texture(texRV, UV_RV);
    vec4 colMVL = texture(texMVL, UV_MVL);
    vec4 colMVR = texture(texMVR, UV_MVR);
    
    // 简单的权重检查 (防止除以0)
    float wTotal = Weights.x + Weights.y + Weights.z + Weights.w;
    if (wTotal < 0.001) {
        FragColor = vec4(0.1, 0.1, 0.1, 1.0); // 无效区域显示深灰
        return;
    }
    
    // 混合
    vec4 finalColor = colFV * Weights.x + colRV * Weights.y + colMVL * Weights.z + colMVR * Weights.w;
    
    // 归一化颜色
    FragColor = finalColor / wTotal;
}
"""

# ==========================================
# PART 3: 预计算 LUT (Geometry Generation)
# ==========================================

def create_bowl_data_lut(cameras, world_range=20.0, rings=80, sectors=128):
    """
    生成 VBO 数据：
    包含：[X, Y, Z] + [U1, V1] + [U2, V2] + [U3, V3] + [U4, V4] + [W1, W2, W3, W4]
    """
    vertices = []
    
    # 碗状参数
    flat_percent = 0.40
    flat_radius = world_range * flat_percent
    curvature = 0.08
    
    # 生成网格点 (极坐标)
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
            grid_pts.append([x, y, z]) # World space: X, Y, Z (Bowl up is Z here, handled later)

    grid_pts = np.array(grid_pts, dtype=np.float32) # (N, 3)
    
    # --- 批量计算 LUT ---
    # 我们有 4 个相机，计算每个点在 4 个相机里的 UV
    # 注意: grid_pts 是 [x, y, z]，假设 Z 是高。
    # Woodscape 世界坐标系定义可能不同，通常 Z 是高。
    # 之前的代码 generate_steep_bowl 用的是 [x, y, z]，投影时用的也是这个。
    
    uv_maps = []
    for cam in cameras:
        # get_uv_for_world_points 需要 [x, y, z]
        uv = cam.get_uv_for_world_points(grid_pts)
        uv_maps.append(uv) # list of (N, 2)
        
    # --- 计算权重 (Weight) ---
    # 使用简单的角度权重或距离权重
    # 这里为了简单，仅使用方位角权重
    weights = np.zeros((len(grid_pts), 4), dtype=np.float32)
    
    # 计算每个点的极角 (-pi, pi)
    angles = np.arctan2(grid_pts[:, 1], grid_pts[:, 0]) # y, x
    
    # 定义每个相机的中心角度 (根据 Woodscape 布局估算)
    # FV(前): 0度 (X正?), RV(后): 180, MVL(左): 90, MVR(右): -90
    # 注意 Woodscape 的 X/Y 轴向。根据之前结果:
    # FV 对应 +X? 还是 +Y? 
    # 假设标准车身坐标: X前, Y左, Z上。
    
    # 简单权重逻辑：
    # 计算点是否在该相机的视野内 (UV在 0-1 之间)
    for k in range(4):
        u = uv_maps[k][:, 0]
        v = uv_maps[k][:, 1]
        valid = (u > 0) & (u < 1) & (v > 0) & (v < 1)
        
        # 边缘淡出 (Soft blending)
        dist_edge = np.minimum(np.minimum(u, 1-u), np.minimum(v, 1-v))
        w = np.maximum(0, dist_edge)
        w = np.power(w, 0.5) # 调整融合硬度
        
        weights[:, k] = w * valid.astype(np.float32)

    # 归一化权重 (在 Shader 里做也行，但这里做更安全)
    # w_sum = np.sum(weights, axis=1, keepdims=True) + 1e-6
    # weights /= w_sum
    
    # --- 组装 VBO 数据 ---
    # Interleaved: Pos(3) + UV_FV(2) + UV_RV(2) + UV_MVL(2) + UV_MVR(2) + Weights(4) = 15 floats per vertex
    
    # OpenGL Y-up 调整: World Z -> GL Y
    gl_pos = np.stack([grid_pts[:, 0], grid_pts[:, 2], -grid_pts[:, 1]], axis=1) # 试探性旋转
    # 或者保持 [x, z, y] 
    gl_pos = np.stack([grid_pts[:, 0], grid_pts[:, 2], grid_pts[:, 1]], axis=1)

    combined_data = []
    
    # 生成索引 (IBO)
    indices = []
    for i in range(rings):
        for j in range(sectors):
            curr = i * sectors + j
            next_row = (i+1) * sectors + j
            curr_next = i * sectors + (j+1)%sectors
            next_row_next = (i+1) * sectors + (j+1)%sectors
            
            indices.extend([curr, next_row, curr_next])
            indices.extend([curr_next, next_row, next_row_next])

    # 打包顶点数据
    # 这是一个巨大的数组拼接
    vertex_data = np.hstack([
        gl_pos,          # 0: Pos
        uv_maps[0],      # 1: FV
        uv_maps[1],      # 2: RV
        uv_maps[2],      # 3: MVL
        uv_maps[3],      # 4: MVR
        weights          # 5: W
    ]).flatten().astype(np.float32)
    
    return vertex_data, np.array(indices, dtype=np.uint32)

# ==========================================
# PART 4: 主程序 V33
# ==========================================

def main():
    if not glfw.init(): return
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.SAMPLES, 4)
    
    window = glfw.create_window(1024, 768, "V33: 3D AVM Shader Direct Rendering", None, None)
    if not window: glfw.terminate(); return
    glfw.make_context_current(window)
    
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_MULTISAMPLE)

    # 1. 编译 Shader
    shader = compileProgram(
        compileShader(VERTEX_SHADER, GL_VERTEX_SHADER),
        compileShader(FRAGMENT_SHADER, GL_FRAGMENT_SHADER)
    )
    glUseProgram(shader)

    # 2. 加载相机和图片
    print("Loading Cameras & Images...")
    cam_files = ['./calib/cam3.json', './calib/cam0.json', './calib/cam1.json', './calib/cam2.json'] # FV, RV, MVL, MVR
    img_files = ['./imgs_raw/cam3.png', './imgs_raw/cam0.png', './imgs_raw/cam1.png', './imgs_raw/cam2.png']
    
    cameras = []
    textures = []
    
    for i, (cf, imf) in enumerate(zip(cam_files, img_files)):
        if not os.path.exists(cf): continue
        cameras.append(Camera(cf, f"Cam{i}"))
        
        # 加载纹理
        img = cv2.imread(imf)
        img = cv2.flip(img, 0) # Flip for OpenGL
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        tid = glGenTextures(1)
        glActiveTexture(GL_TEXTURE0 + i) # 绑定到 unit 0, 1, 2, 3
        glBindTexture(GL_TEXTURE_2D, tid)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, img.shape[1], img.shape[0], 0, GL_RGB, GL_UNSIGNED_BYTE, img)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR) # No Mipmap for now
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        textures.append(tid)
        
        # 设置 Shader 的 sampler uniform
        loc = glGetUniformLocation(shader, ["texFV", "texRV", "texMVL", "texMVR"][i])
        glUniform1i(loc, i)

    # 3. 生成网格数据 (LUT)
    print("Generating 3D Bowl LUT...")
    v_data, i_data = create_bowl_data_lut(cameras, world_range=18.0)
    
    # 4. 上传 VBO/VAO
    vao = glGenVertexArrays(1)
    glBindVertexArray(vao)
    
    vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, v_data.nbytes, v_data, GL_STATIC_DRAW)
    
    ebo = glGenBuffers(1)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, i_data.nbytes, i_data, GL_STATIC_DRAW)
    
    # 设置属性指针 (Stride = 15 floats * 4 bytes = 60)
    stride = 15 * 4
    # 0: Pos (3)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
    glEnableVertexAttribArray(0)
    # 1: UV_FV (2) -> offset 3*4 = 12
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(12))
    glEnableVertexAttribArray(1)
    # 2: UV_RV (2) -> offset 5*4 = 20
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(20))
    glEnableVertexAttribArray(2)
    # 3: UV_MVL (2) -> offset 7*4 = 28
    glVertexAttribPointer(3, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(28))
    glEnableVertexAttribArray(3)
    # 4: UV_MVR (2) -> offset 9*4 = 36
    glVertexAttribPointer(4, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(36))
    glEnableVertexAttribArray(4)
    # 5: Weights (4) -> offset 11*4 = 44
    glVertexAttribPointer(5, 4, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(44))
    glEnableVertexAttribArray(5)

    # MVP locations
    model_loc = glGetUniformLocation(shader, "model")
    view_loc = glGetUniformLocation(shader, "view")
    proj_loc = glGetUniformLocation(shader, "projection")

    # 交互变量
    rot_x, rot_y = 45, 0
    zoom = -25.0
    last_x, last_y = 0, 0
    is_dragging = False

    def mouse_cb(w, b, a, m):
        nonlocal is_dragging, last_x, last_y
        if b == glfw.MOUSE_BUTTON_LEFT:
            if a == glfw.PRESS:
                is_dragging = True
                last_x, last_y = glfw.get_cursor_pos(w)
            elif a == glfw.RELEASE:
                is_dragging = False
    
    def cursor_cb(w, x, y):
        nonlocal rot_x, rot_y, last_x, last_y
        if is_dragging:
            rot_y += (x - last_x) * 0.5
            rot_x += (y - last_y) * 0.5
            last_x, last_y = x, y
            
    def scroll_cb(w, x, y):
        nonlocal zoom
        zoom += y * 1.0

    glfw.set_mouse_button_callback(window, mouse_cb)
    glfw.set_cursor_pos_callback(window, cursor_cb)
    glfw.set_scroll_callback(window, scroll_cb)

    print("Rendering... Drag to rotate.")
    
    while not glfw.window_should_close(window):
        w, h = glfw.get_framebuffer_size(window)
        glViewport(0, 0, w, h)
        glClearColor(0.1, 0.1, 0.1, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        # Projection
        aspect = w / h if h > 0 else 1.0
        # 手动构建投影矩阵或使用库 (这里简化，假设 gluPerspective 等效)
        glMatrixMode(GL_PROJECTION); glLoadIdentity(); gluPerspective(45, aspect, 0.1, 100.0)
        proj_mat = glGetFloatv(GL_PROJECTION_MATRIX) # Hack: get from fixed pipeline
        
        # View
        glMatrixMode(GL_MODELVIEW); glLoadIdentity(); glTranslatef(0, -5, zoom)
        glRotatef(rot_x, 1, 0, 0); glRotatef(rot_y, 0, 1, 0)
        view_mat = glGetFloatv(GL_MODELVIEW_MATRIX)
        
        # Model
        model_mat = np.eye(4, dtype=np.float32)

        # Upload Matrices
        glUniformMatrix4fv(proj_loc, 1, GL_FALSE, proj_mat)
        glUniformMatrix4fv(view_loc, 1, GL_FALSE, view_mat)
        glUniformMatrix4fv(model_loc, 1, GL_TRUE, model_mat) # transpose=True for row-major numpy

        # Draw Bowl
        glBindVertexArray(vao)
        glDrawElements(GL_TRIANGLES, len(i_data), GL_UNSIGNED_INT, None)
        
        # Draw Car (Placeholder)
        # ... skipped for simplicity ...

        glfw.swap_buffers(window)
        glfw.poll_events()

    glfw.terminate()

if __name__ == "__main__":
    main()