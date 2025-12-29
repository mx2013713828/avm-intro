import numpy as np
import cv2
import os

def load_roecs_config(yaml_path):
    print(f"Loading config from {yaml_path}")
    fs = cv2.FileStorage(yaml_path, cv2.FILE_STORAGE_READ)
    
    cameras = ["front", "back", "left", "right"]
    params = {}
    
    for cam in cameras:
        params[cam] = {
            "K": fs.getNode(f"{cam}_camera_matrix").mat(),
            "D": fs.getNode(f"{cam}_dist_coeffs").mat(),
            "T": fs.getNode(f"{cam}_extrinsic").mat(),
            "res": fs.getNode(f"{cam}_resolution").mat().flatten().astype(int)
        }
    fs.release()
    return params

def generate_lut():
    config_path = "../config/roecs.yaml"
    if not os.path.exists(config_path):
        config_path = "config/roecs.yaml"
        
    params = load_roecs_config(config_path)
    
    # BEV parameters: 1000x1000, 1cm/pixel (0.1dm/pixel)
    W, H = 1000, 1000
    dx = 0.1
    
    # World grid
    y_idx, x_idx = np.indices((H, W), dtype=np.float32)
    # Center (500, 500) is world (0, 0). 
    # NOTE: Front is Y+, Right is X+ in many systems. 
    # Let's check the user's T_FG logic: t_FG = [0.67, 25.08, 3.17]. 
    # Y=25.08 dm (2.5m) is forward. So Front is indeed Y+.
    Xw = (x_idx - W/2) * dx
    Yw = (H/2 - y_idx) * dx
    Zw = np.zeros_like(Xw)
    
    # 3D points in world (N, 3)
    pts_world = np.stack([Xw, Yw, Zw], axis=-1).reshape(-1, 3)
    
    camera_coords = {}
    masks = {}
    
    for cam in ["front", "back", "left", "right"]:
        print(f"Projecting {cam} camera...")
        p = params[cam]
        # P_cam = R * P_world + t
        # extrinsic is 4x4: [R t; 0 1]
        R = p["T"][:3, :3]
        t = p["T"][:3, 3]
        
        pts_cam = (pts_world @ R.T) + t
        
        # Project using Fisheye model
        # cv2.fisheye.projectPoints expects (N, 1, 3)
        pts_2d, _ = cv2.fisheye.projectPoints(
            pts_cam.reshape(-1, 1, 3), 
            np.zeros(3), np.zeros(3), 
            p["K"], p["D"]
        )
        pts_2d = pts_2d.reshape(H, W, 2)
        
        # Check if points are in front of camera (Z_cam > 0)
        z_mask = (pts_cam[:, 2].reshape(H, W) > 0)
        
        # Check if points are within image resolution
        res_w, res_h = p["res"]
        image_mask = (pts_2d[..., 0] >= 0) & (pts_2d[..., 0] < res_w) & \
                     (pts_2d[..., 1] >= 0) & (pts_2d[..., 1] < res_h)
        
        masks[cam] = z_mask & image_mask
        camera_coords[cam] = pts_2d

    # --- Blending and Feathering Optimization V2 ---
    # 1. Calculate Edge Weights (Feathering) for each camera
    edge_weights = {}
    for cam in ["front", "back", "left", "right"]:
        res_w, res_h = params[cam]["res"]
        coords = camera_coords[cam]
        # Distance to closest edge
        dx_edge = np.minimum(coords[..., 0], res_w - coords[..., 0])
        dy_edge = np.minimum(coords[..., 1], res_h - coords[..., 1])
        d_edge = np.minimum(dx_edge, dy_edge)
        
        # INCREASED Feathering width: 80 pixels for a very soft fade
        feather_w = 80.0
        w = np.clip(d_edge / feather_w, 0.0, 1.0)
        # Smooth step
        w = 0.5 * (1 - np.cos(np.pi * w))
        edge_weights[cam] = w * masks[cam]

    # 2. Smooth Angular Blending
    angles = np.arctan2(Yw, Xw)
    # INCREASED: 30 degrees for extremely smooth blending
    blend_width = np.deg2rad(30) 
    
    def get_smooth_weight(angle, target_angle):
        diff = angle - target_angle
        # Normalize diff to [-pi, pi]
        diff = (diff + np.pi) % (2 * np.pi) - np.pi
        # Map blend zone to [0, 1]
        t = (diff / blend_width) + 0.5
        t = np.clip(t, 0.0, 1.0)
        # Periodic Sine smooth: 0.5 + 0.5 * np.sin(np.pi * (t - 0.5))
        return 0.5 + 0.5 * np.sin(np.pi * (t - 0.5))

    # Define quadrant masks
    front_mask = (angles >= np.pi/4) & (angles < 3*np.pi/4)
    left_mask = (angles >= 3*np.pi/4) | (angles < -3*np.pi/4)
    back_mask = (angles >= -3*np.pi/4) & (angles < -np.pi/4)
    right_mask = (angles >= -np.pi/4) & (angles < np.pi/4)
    
    # Define blend indices
    # 5: Front & Right (pi/4)
    # 6: Front & Left (3pi/4)
    # 4: Back & Left (-3pi/4)
    # 7: Back & Right (-pi/4)
    is_5 = (angles > np.pi * 0.25 - blend_width/2) & (angles < np.pi * 0.25 + blend_width/2)
    is_6 = (angles > np.pi * 0.75 - blend_width/2) & (angles < np.pi * 0.75 + blend_width/2)
    is_4 = (angles > -np.pi * 0.75 - blend_width/2) & (angles < -np.pi * 0.75 + blend_width/2)
    is_7 = (angles > -np.pi * 0.25 - blend_width/2) & (angles < -np.pi * 0.25 + blend_width/2)

    flags = np.full((H, W), -1, dtype=np.float32)
    weights = np.ones((H, W), dtype=np.float32)

    # Assign single camera regions
    flags[front_mask] = 0
    flags[left_mask] = 1
    flags[back_mask] = 2
    flags[right_mask] = 3
    
    # Assign blend zones and calculate weights
    # Note: Weight W in LUT means: Final = CamA * (1-W) + CamB * W
    
    # Front(0) & Right(3) -> 5. Boundary at pi/4. W=0 is Right, W=1 is Front
    # Combined with feathering: W = (W_smooth * E_0) / (W_smooth * E_0 + (1-W_smooth) * E_3)
    if np.any(is_5):
        w_s = get_smooth_weight(angles[is_5], np.pi * 0.25)
        e0 = edge_weights["front"][is_5]
        e3 = edge_weights["right"][is_5]
        weights[is_5] = (w_s * e0) / (w_s * e0 + (1-w_s) * e3 + 1e-6)
        flags[is_5] = 5
    
    # Front(0) & Left(1) -> 6. Boundary at 3pi/4. W=0 is Front, W=1 is Left
    if np.any(is_6):
        w_s = get_smooth_weight(angles[is_6], np.pi * 0.75)
        e0 = edge_weights["front"][is_6]
        e1 = edge_weights["left"][is_6]
        weights[is_6] = ((1-w_s) * e0) / ((1-w_s) * e0 + w_s * e1 + 1e-6)
        # Note: the kernel 'blend' function does: A * weight + B * (1-weight)
        # For flag 6: A=0(Front), B=1(Left). So weight=1-w_final
        flags[is_6] = 6
    
    # Back(2) & Left(1) -> 4. Boundary at -3pi/4. W=0 is Left, W=1 is Back
    if np.any(is_4):
        w_s = get_smooth_weight(angles[is_4], -np.pi * 0.75)
        e2 = edge_weights["back"][is_4]
        e1 = edge_weights["left"][is_4]
        weights[is_4] = (w_s * e2) / (w_s * e2 + (1-w_s) * e1 + 1e-6)
        flags[is_4] = 4
    
    # Back(2) & Right(3) -> 7. Boundary at -pi/4. W=0 is Back, W=1 is Right
    if np.any(is_7):
        w_s = get_smooth_weight(angles[is_7], -np.pi * 0.25)
        e2 = edge_weights["back"][is_7]
        e3 = edge_weights["right"][is_7]
        weights[is_7] = ((1-w_s) * e2) / ((1-w_s) * e2 + w_s * e3 + 1e-6)
        flags[is_7] = 7

    # Final table construction
    # table = [flags, weight_map, c0_x, c0_y, c1_x, c1_y, c2_x, c2_y, c3_x, c3_y] (10 floats)
    out_table = np.zeros((H, W, 10), dtype=np.float32)
    out_table[..., 0] = flags
    out_table[..., 1] = weights
    
    cams = ["front", "left", "back", "right"]
    for i, cam in enumerate(cams):
        out_table[..., 2 + i*2] = camera_coords[cam][..., 0]
        out_table[..., 2 + i*2 + 1] = camera_coords[cam][..., 1]
        
    # Validation: set invalid flag where data is missing
    # A point is valid if the assigned camera(s) have data
    final_mask = np.zeros((H, W), dtype=bool)
    for i, cam in enumerate(cams):
        final_mask |= ((flags == i) & masks[cam])
    
    # Blend masks
    final_mask |= (flags == 4) & masks["back"] & masks["left"]
    final_mask |= (flags == 5) & masks["front"] & masks["right"]
    final_mask |= (flags == 6) & masks["front"] & masks["left"]
    final_mask |= (flags == 7) & masks["back"] & masks["right"]
    
    out_table[~final_mask, 0] = -1
    
    # Save
    out_table.tofile("surround_view.binary")
    print(f"LUT generated: surround_view.binary ({H}x{W})")

if __name__ == "__main__":
    generate_lut()
