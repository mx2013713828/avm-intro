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

    # Blending and region assignment
    # We'll use radial angle to assign regions
    angles = np.arctan2(Yw, Xw) # -pi to pi. 0 is Right, pi/2 is Front
    
    # Define blending zones (approx 10 degrees at corners)
    blend_width = np.deg2rad(10)
    
    front_mask = (angles >= np.pi/4) & (angles < 3*np.pi/4)
    left_mask = (angles >= 3*np.pi/4) | (angles < -3*np.pi/4)
    back_mask = (angles >= -3*np.pi/4) & (angles < -np.pi/4)
    right_mask = (angles >= -np.pi/4) & (angles < np.pi/4)

    # Simplified blending zones:
    # 4: back & left (idxs[0] = {2, 1})
    # 5: front & right (idxs[1] = {0, 3})
    # 6: front & left (idxs[2] = {0, 1})
    # 7: back & right (idxs[3] = {2, 3})
    
    flags = np.full((H, W), -1, dtype=np.float32)
    weights = np.zeros((H, W), dtype=np.float32)
    
    # 1. Assign single camera regions (non-blending zones for now)
    # We will refine this with actual blend areas
    
    def get_blend_weight(angle, target_angle):
        # target_angle is the boundary: e.g. pi/4
        diff = angle - target_angle
        # Normalize diff to [-pi, pi]
        diff = (diff + np.pi) % (2 * np.pi) - np.pi
        w = 0.5 + (diff / blend_width) * 0.5
        return np.clip(w, 0, 1)

    # Intersection lines
    # FL: 3pi/4
    # FR: pi/4
    # BL: -3pi/4
    # BR: -pi/4
    
    # Calculate weights for each transition
    # 6: Front & Left (PI * 3/4)
    w_6 = get_blend_weight(angles, np.pi * 0.75)
    is_6 = (angles > np.pi * 0.75 - blend_width/2) & (angles < np.pi * 0.75 + blend_width/2)
    
    # 5: Front & Right (PI * 1/4)
    w_5 = get_blend_weight(angles, np.pi * 0.25)
    is_5 = (angles > np.pi * 0.25 - blend_width/2) & (angles < np.pi * 0.25 + blend_width/2)
    
    # 7: Back & Right (-PI * 1/4)
    w_7 = get_blend_weight(angles, -np.pi * 0.25)
    is_7 = (angles > -np.pi * 0.25 - blend_width/2) & (angles < -np.pi * 0.25 + blend_width/2)

    # 4: Back & Left (-PI * 3/4)
    w_4 = get_blend_weight(angles, -np.pi * 0.75)
    is_4 = (angles > -np.pi * 0.75 - blend_width/2) & (angles < -np.pi * 0.75 + blend_width/2)

    # Assign regions
    flags[front_mask] = 0
    flags[left_mask] = 1
    flags[back_mask] = 2
    flags[right_mask] = 3
    
    # Apply blends
    flags[is_6] = 6
    weights[is_6] = w_6[is_6]
    
    flags[is_5] = 5
    weights[is_5] = w_5[is_5]
    
    flags[is_7] = 7
    weights[is_7] = w_7[is_7]
    
    flags[is_4] = 4
    weights[is_4] = w_4[is_4]

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
