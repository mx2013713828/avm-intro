import sys
import os
import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn.functional as F
from collections import namedtuple

# Import from local surround_utils
try:
    import surround_utils
except ImportError:
    # If running from parent dir, add stitching to path
    sys.path.append(os.path.join(os.getcwd(), "stitching"))
    import surround_utils

def generate_weights():
    print("Generating weights.png...")
    names = surround_utils.camera_names
    
    # Paths relative to this script (assuming script is in stitching/)
    # But we might run it from project root. Let's handle both.
    
    # If running from stitching/
    if os.path.basename(os.getcwd()) == "stitching":
        base_dir = ".."
    else:
        base_dir = "."
        
    images_dir = os.path.join(base_dir, "images")
    yaml_dir = os.path.join(base_dir, "yaml")
    
    images = [os.path.join(images_dir, name + ".png") for name in names]
    yamls = [os.path.join(yaml_dir, name + ".yaml") for name in names]
    
    camera_models = [surround_utils.FisheyeCameraModel(camera_file, camera_name) for camera_file, camera_name in zip(yamls, names)]

    projected = []
    for image_file, camera in zip(images, camera_models):
        img = cv2.imread(image_file)
        if img is None:
            print(f"Failed to load image: {image_file}")
            continue
        img = camera.undistort(img)
        img = camera.project(img)
        img = camera.flip(img)
        projected.append(img)

    birdview = surround_utils.BirdView()
    Gmat, Mmat = birdview.get_weights_and_masks(projected)
    birdview.update_frames(projected)
    birdview.make_luminance_balance().stitch_all_parts()
    birdview.make_white_balance()
    birdview.copy_car_image()
    
    # Save weights.png and masks.png in the current directory (stitching/ or root)
    # The user asked to put generate_data.py in stitching folder.
    # If we run it, we probably want the output in the same folder or where cuda.cu expects it.
    # cuda.cu expects surround_view.binary.
    
    Image.fromarray((Gmat * 255).astype(np.uint8)).save("weights.png")
    Image.fromarray(Mmat.astype(np.uint8)).save("masks.png")
    print("Generated weights.png and masks.png")

def load_camera_param(camera):
    # Determine base dir
    if os.path.basename(os.getcwd()) == "stitching":
        base_dir = ".."
    else:
        base_dir = "."
        
    yaml_path = os.path.join(base_dir, "yaml", f"{camera}.yaml")
    
    fs = cv2.FileStorage(yaml_path, cv2.FILE_STORAGE_READ)
    camera_matrix = fs.getNode("camera_matrix").mat()
    dist_coeffs = fs.getNode("dist_coeffs").mat()
    resolution = fs.getNode("resolution").mat().reshape(2).tolist()
    project_matrix = fs.getNode("project_matrix").mat()
    scale_xy = fs.getNode("scale_xy").mat().reshape(2).tolist()
    shift_xy = fs.getNode("shift_xy").mat().reshape(2).tolist()
    
    shapes = {
        "front": [1200, 550],
        "back": [1200, 550],
        "left": [1600, 500],
        "right": [1600, 500],
    }
    
    oris = {
        "front": "n",
        "left": "r-",
        "back": "m",
        "right": "r+"
    }
    
    CameraParam = namedtuple("CameraParam", ["camera_matrix", "dist_coeffs", "resolution", "project", "scale", "shift",  "shape", "ori"])
    return CameraParam(camera_matrix, dist_coeffs, resolution, project_matrix, scale_xy, shift_xy, shapes[camera], oris[camera])


def get_rotation_matrix(theta, w, h):
    horizontal = theta % 180 == 0
    theta = theta / 180 * np.pi
    R90 = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])
    
    if horizontal:
        To = np.array([
            [1, 0, -w*0.5],
            [0, 1, -h*0.5],
            [0, 0, 1]
        ])

        Tn = np.array([
            [1, 0, w*0.5],
            [0, 1, h*0.5],
            [0, 0, 1]
        ])
        return Tn @ R90 @ To
    else:
        To = np.array([
            [1, 0, -w*0.5],
            [0, 1, -h*0.5],
            [0, 0, 1]
        ])

        Tn = np.array([
            [1, 0, h*0.5],
            [0, 1, w*0.5],
            [0, 0, 1]
        ])
        return Tn @ R90 @ To

def rotation(operator, shape):
    if operator == "r+":
        return get_rotation_matrix(90, shape[0], shape[1]), shape[::-1]
    elif operator == "r-":
        return get_rotation_matrix(-90, shape[0], shape[1]), shape[::-1]
    elif operator == "m":
        return get_rotation_matrix(180, shape[0], shape[1]), shape
    return np.eye(3), shape

def generate_binary():
    print("Generating surround_view.binary...")
    cameras = ["front", "left", "back", "right"]
    camera_params = {}
    images = {}
    
    if os.path.basename(os.getcwd()) == "stitching":
        base_dir = ".."
    else:
        base_dir = "."
        
    for camera in cameras:
        camera_params[camera] = load_camera_param(camera) 
        images[camera] = cv2.imread(os.path.join(base_dir, "images", f"{camera}.png"))
        
    if not os.path.exists("weights.png"):
        print("weights.png not found!")
        return
        
    weights = cv2.imread("weights.png", -1) / 255
    
    SW = 1200
    SH = 1600
    
    table = []
    for camera in cameras:
        image = images[camera]
        param = camera_params[camera]
        
        scaled_camera_matrix = param.camera_matrix.copy()
        scaled_camera_matrix[[0, 1], [0, 1]] *= param.scale
        scaled_camera_matrix[:2, 2] += param.shift
        x, y = cv2.fisheye.initUndistortRectifyMap(param.camera_matrix, param.dist_coeffs, None, scaled_camera_matrix, param.resolution, cv2.CV_32F)
        undistorted_image = cv2.remap(image, x, y, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)
        IH, IW = undistorted_image.shape[:2]
        proj, shape = rotation(param.ori, param.shape)
        w, h = shape
        
        ys, xs = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
        
        xy = torch.stack([xs + 0.5, ys + 0.5, torch.ones_like(xs)], dim=-1).float()
    
        xy = xy @ torch.from_numpy(proj @ param.project).float().inverse().T
        xy[..., :2] /= xy[..., [2]]
        xy = ((xy[..., :2] - 0.5) / torch.tensor([IW, IH]))[None] * 2 - 1
        remapxy = torch.stack([torch.from_numpy(x) + 0.5, torch.from_numpy(y) + 0.5], dim=-1)
        output = F.grid_sample(remapxy.permute(2, 0, 1)[None], xy.float(), "bilinear", align_corners=False)[0].permute(1, 2, 0)
        
        if camera == "left":
            cx = w
            cy = h / 2
            M = np.array([
                [1, 0, -cx + w],
                [0, 1, -cy + SH / 2]
            ])
        elif camera == "right":
            cx = 0
            cy = h / 2
            M = np.array([
                [1, 0, -cx + SW - w],
                [0, 1, -cy + SH / 2]
            ])
        elif camera == "front":
            cx = w / 2
            cy = h
            M = np.array([
                [1, 0, -cx + SW / 2],
                [0, 1, -cy + h]
            ])
        elif camera == "back":
            cx = w / 2
            cy = 0
            M = np.array([
                [1, 0, -cx + SW / 2],
                [0, 1, -cy + SH - h]
            ])
        output_coord = cv2.warpAffine(output.data.numpy(), M, dsize=(SW, SH), flags=cv2.INTER_NEAREST)
        output_coord[..., 0] = output_coord[..., 0].clip(0, IW-1)
        output_coord[..., 1] = output_coord[..., 1].clip(0, IH-1)
        table.append(output_coord)

    front, left, back, right = table
    
    flags = np.full((SH, SW, 1), -1)  
    flags[front[..., 0] != 0] = 0
    flags[left[..., 0] != 0]  = 1
    flags[back[..., 0] != 0]  = 2
    flags[right[..., 0] != 0] = 3
    weight_map = np.zeros((SH, SW, 1))
    for i in range(4):
        
        weight = weights[..., i]
        wH, wW = weight.shape[:2]
        
        x, y = [
            [0, SH - wH],
            [SW - wW, 0],
            [0, 0],
            [SW - wW, SH - wH]
        ][i]
        flags[y:y+wH, x:x+wW, 0] = [
            4,  # 4 -> 2, 1
            5,  # 5 -> 0, 3
            6,  # 6 -> 0, 1
            7,  # 7 -> 2, 3
        ][i]
        weight_map[y:y+wH, x:x+wW, 0] = weight
        
    table = np.concatenate([flags, weight_map] + table, axis=-1)
    table.astype(np.float32).tofile("surround_view.binary")
    print("Generated surround_view.binary")

if __name__ == "__main__":
    generate_weights()
    generate_binary()
