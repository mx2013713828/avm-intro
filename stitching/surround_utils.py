import os
import cv2
import numpy as np
from PIL import Image

# ============================================================================
# Parameter Settings
# ============================================================================

camera_names = ["front", "back", "left", "right"]

# (shift_width, shift_height): how far away the birdview looks outside
# of the calibration pattern in horizontal and vertical directions
shift_w = 300
shift_h = 300

# size of the gap between the calibration pattern and the car
# in horizontal and vertical directions
inn_shift_w = 20
inn_shift_h = 50

# total width/height of the stitched image
total_w = 600 + 2 * shift_w
total_h = 1000 + 2 * shift_h

# four corners of the rectangular region occupied by the car
# top-left (x_left, y_top), bottom-right (x_right, y_bottom)
xl = shift_w + 180 + inn_shift_w
xr = total_w - xl
yt = shift_h + 200 + inn_shift_h
yb = total_h - yt

project_shapes = {
    "front": (total_w, yt),
    "back":  (total_w, yt),
    "left":  (total_h, xl),
    "right": (total_h, xl)
}

# Car image path - assuming running from stitching folder, so ../images
# But we will handle path in the class or pass it in
CAR_IMAGE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "images", "car.png")

if os.path.exists(CAR_IMAGE_PATH):
    car_image = cv2.imread(CAR_IMAGE_PATH)
    if car_image is not None:
        car_image = cv2.resize(car_image, (xr - xl, yb - yt))
else:
    # Fallback or warning
    print(f"Warning: Car image not found at {CAR_IMAGE_PATH}")
    car_image = np.zeros((yb - yt, xr - xl, 3), dtype=np.uint8)


# ============================================================================
# Utils
# ============================================================================

def convert_binary_to_bool(mask):
    return (mask.astype(float) / 255.0).astype(int)

def adjust_luminance(gray, factor):
    return np.minimum((gray * factor), 255).astype(np.uint8)

def get_mean_statistisc(gray, mask):
    return np.sum(gray * mask)

def mean_luminance_ratio(grayA, grayB, mask):
    return get_mean_statistisc(grayA, mask) / get_mean_statistisc(grayB, mask)

def get_mask(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
    return mask

def get_overlap_region_mask(imA, imB):
    overlap = cv2.bitwise_and(imA, imB)
    mask = get_mask(overlap)
    mask = cv2.dilate(mask, np.ones((2, 2), np.uint8), iterations=2)
    return mask

def get_outmost_polygon_boundary(img):
    mask = get_mask(img)
    mask = cv2.dilate(mask, np.ones((2, 2), np.uint8), iterations=2)
    cnts, hierarchy = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)[-2:]
    C = sorted(cnts, key=lambda x: cv2.contourArea(x), reverse=True)[0]
    polygon = cv2.approxPolyDP(C, 0.009 * cv2.arcLength(C, True), True)
    return polygon

def get_weight_mask_matrix(imA, imB, dist_threshold=5):
    overlapMask = get_overlap_region_mask(imA, imB)
    overlapMaskInv = cv2.bitwise_not(overlapMask)
    indices = np.where(overlapMask == 255)

    imA_diff = cv2.bitwise_and(imA, imA, mask=overlapMaskInv)
    imB_diff = cv2.bitwise_and(imB, imB, mask=overlapMaskInv)

    G = get_mask(imA).astype(np.float32) / 255.0

    polyA = get_outmost_polygon_boundary(imA_diff)
    polyB = get_outmost_polygon_boundary(imB_diff)

    for y, x in zip(*indices):
        xy_tuple = tuple([int(x), int(y)])
        distToB = cv2.pointPolygonTest(polyB, xy_tuple, True)
        if distToB < dist_threshold:
            distToA = cv2.pointPolygonTest(polyA, xy_tuple, True)
            distToB *= distToB
            distToA *= distToA
            G[y, x] = distToB / (distToA + distToB)

    return G, overlapMask

def make_white_balance(image):
    B, G, R = cv2.split(image)
    m1 = np.mean(B)
    m2 = np.mean(G)
    m3 = np.mean(R)
    K = (m1 + m2 + m3) / 3
    c1 = K / m1
    c2 = K / m2
    c3 = K / m3
    B = adjust_luminance(B, c1)
    G = adjust_luminance(G, c2)
    R = adjust_luminance(R, c3)
    return cv2.merge((B, G, R))


# ============================================================================
# Fisheye Camera Model
# ============================================================================

class FisheyeCameraModel(object):
    def __init__(self, camera_param_file, camera_name):
        if not os.path.isfile(camera_param_file):
            raise ValueError(f"Cannot find camera param file: {camera_param_file}")

        if camera_name not in camera_names:
            raise ValueError(f"Unknown camera name: {camera_name}")

        self.camera_file = camera_param_file
        self.camera_name = camera_name
        self.scale_xy = (1.0, 1.0)
        self.shift_xy = (0, 0)
        self.undistort_maps = None
        self.project_matrix = None
        self.project_shape = project_shapes[self.camera_name]
        self.load_camera_params()

    def load_camera_params(self):
        fs = cv2.FileStorage(self.camera_file, cv2.FILE_STORAGE_READ)
        self.camera_matrix = fs.getNode("camera_matrix").mat()
        self.dist_coeffs = fs.getNode("dist_coeffs").mat()
        self.resolution = fs.getNode("resolution").mat().flatten()

        scale_xy = fs.getNode("scale_xy").mat()
        if scale_xy is not None:
            self.scale_xy = scale_xy

        shift_xy = fs.getNode("shift_xy").mat()
        if shift_xy is not None:
            self.shift_xy = shift_xy

        project_matrix = fs.getNode("project_matrix").mat()
        if project_matrix is not None:
            self.project_matrix = project_matrix

        fs.release()
        self.update_undistort_maps()

    def update_undistort_maps(self):
        new_matrix = self.camera_matrix.copy()
        new_matrix[0, 0] *= self.scale_xy[0]
        new_matrix[1, 1] *= self.scale_xy[1]
        new_matrix[0, 2] += self.shift_xy[0]
        new_matrix[1, 2] += self.shift_xy[1]
        width, height = self.resolution

        self.undistort_maps = cv2.fisheye.initUndistortRectifyMap(
            self.camera_matrix,
            self.dist_coeffs,
            np.eye(3),
            new_matrix,
            (int(width), int(height)),
            cv2.CV_16SC2
        )
        return self

    def undistort(self, image):
        result = cv2.remap(image, *self.undistort_maps, interpolation=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT)
        return result

    def project(self, image):
        result = cv2.warpPerspective(image, self.project_matrix, self.project_shape)
        return result

    def flip(self, image):
        if self.camera_name == "front":
            return image.copy()
        elif self.camera_name == "back":
            return image.copy()[::-1, ::-1, :]
        elif self.camera_name == "left":
            return cv2.transpose(image)[::-1]
        else:
            return np.flip(cv2.transpose(image), 1)


# ============================================================================
# BirdView
# ============================================================================

def FI(front_image): return front_image[:, :xl]
def FII(front_image): return front_image[:, xr:]
def FM(front_image): return front_image[:, xl:xr]
def BIII(back_image): return back_image[:, :xl]
def BIV(back_image): return back_image[:, xr:]
def BM(back_image): return back_image[:, xl:xr]
def LI(left_image): return left_image[:yt, :]
def LIII(left_image): return left_image[yb:, :]
def LM(left_image): return left_image[yt:yb, :]
def RII(right_image): return right_image[:yt, :]
def RIV(right_image): return right_image[yb:, :]
def RM(right_image): return right_image[yt:yb, :]

class BirdView(object):
    def __init__(self):
        self.image = np.zeros((total_h, total_w, 3), np.uint8)
        self.weights = None
        self.masks = None
        self.car_image = car_image
        self.frames = None

    def update_frames(self, images):
        self.frames = images

    def merge(self, imA, imB, k):
        G = self.weights[k]
        return (imA * G + imB * (1 - G)).astype(np.uint8)

    @property
    def FL(self): return self.image[:yt, :xl]
    @property
    def F(self): return self.image[:yt, xl:xr]
    @property
    def FR(self): return self.image[:yt, xr:]
    @property
    def BL(self): return self.image[yb:, :xl]
    @property
    def B(self): return self.image[yb:, xl:xr]
    @property
    def BR(self): return self.image[yb:, xr:]
    @property
    def L(self): return self.image[yt:yb, :xl]
    @property
    def R(self): return self.image[yt:yb, xr:]
    @property
    def C(self): return self.image[yt:yb, xl:xr]

    def stitch_all_parts(self):
        front, back, left, right = self.frames
        np.copyto(self.F, FM(front))
        np.copyto(self.B, BM(back))
        np.copyto(self.L, LM(left))
        np.copyto(self.R, RM(right))
        np.copyto(self.FL, self.merge(FI(front), LI(left), 0))
        np.copyto(self.FR, self.merge(FII(front), RII(right), 1))
        np.copyto(self.BL, self.merge(BIII(back), LIII(left), 2))
        np.copyto(self.BR, self.merge(BIV(back), RIV(right), 3))

    def copy_car_image(self):
        np.copyto(self.C, self.car_image)

    def make_luminance_balance(self):
        def tune(x):
            if x >= 1:
                return x * np.exp((1 - x) * 0.5)
            else:
                return x * np.exp((1 - x) * 0.8)

        front, back, left, right = self.frames
        # Ensure masks are available
        if self.masks is None:
            print("Masks not calculated yet!")
            return self
            
        m1, m2, m3, m4 = self.masks
        Fb, Fg, Fr = cv2.split(front)
        Bb, Bg, Br = cv2.split(back)
        Lb, Lg, Lr = cv2.split(left)
        Rb, Rg, Rr = cv2.split(right)

        a1 = mean_luminance_ratio(RII(Rb), FII(Fb), m2)
        a2 = mean_luminance_ratio(RII(Rg), FII(Fg), m2)
        a3 = mean_luminance_ratio(RII(Rr), FII(Fr), m2)

        b1 = mean_luminance_ratio(BIV(Bb), RIV(Rb), m4)
        b2 = mean_luminance_ratio(BIV(Bg), RIV(Rg), m4)
        b3 = mean_luminance_ratio(BIV(Br), RIV(Rr), m4)

        c1 = mean_luminance_ratio(LIII(Lb), BIII(Bb), m3)
        c2 = mean_luminance_ratio(LIII(Lg), BIII(Bg), m3)
        c3 = mean_luminance_ratio(LIII(Lr), BIII(Br), m3)

        d1 = mean_luminance_ratio(FI(Fb), LI(Lb), m1)
        d2 = mean_luminance_ratio(FI(Fg), LI(Lg), m1)
        d3 = mean_luminance_ratio(FI(Fr), LI(Lr), m1)

        t1 = (a1 * b1 * c1 * d1)**0.25
        t2 = (a2 * b2 * c2 * d2)**0.25
        t3 = (a3 * b3 * c3 * d3)**0.25

        x1 = tune(t1 / (d1 / a1)**0.5)
        x2 = tune(t2 / (d2 / a2)**0.5)
        x3 = tune(t3 / (d3 / a3)**0.5)

        Fb = adjust_luminance(Fb, x1)
        Fg = adjust_luminance(Fg, x2)
        Fr = adjust_luminance(Fr, x3)

        y1 = tune(t1 / (b1 / c1)**0.5)
        y2 = tune(t2 / (b2 / c2)**0.5)
        y3 = tune(t3 / (b3 / c3)**0.5)

        Bb = adjust_luminance(Bb, y1)
        Bg = adjust_luminance(Bg, y2)
        Br = adjust_luminance(Br, y3)

        z1 = tune(t1 / (c1 / d1)**0.5)
        z2 = tune(t2 / (c2 / d2)**0.5)
        z3 = tune(t3 / (c3 / d3)**0.5)

        Lb = adjust_luminance(Lb, z1)
        Lg = adjust_luminance(Lg, z2)
        Lr = adjust_luminance(Lr, z3)

        w1 = tune(t1 / (a1 / b1)**0.5)
        w2 = tune(t2 / (a2 / b2)**0.5)
        w3 = tune(t3 / (a3 / b3)**0.5)

        Rb = adjust_luminance(Rb, w1)
        Rg = adjust_luminance(Rg, w2)
        Rr = adjust_luminance(Rr, w3)

        self.frames = [cv2.merge((Fb, Fg, Fr)),
                       cv2.merge((Bb, Bg, Br)),
                       cv2.merge((Lb, Lg, Lr)),
                       cv2.merge((Rb, Rg, Rr))]
        return self

    def get_weights_and_masks(self, images):
        front, back, left, right = images
        G0, M0 = get_weight_mask_matrix(FI(front), LI(left))
        G1, M1 = get_weight_mask_matrix(FII(front), RII(right))
        G2, M2 = get_weight_mask_matrix(BIII(back), LIII(left))
        G3, M3 = get_weight_mask_matrix(BIV(back), RIV(right))
        self.weights = [np.stack((G, G, G), axis=2) for G in (G0, G1, G2, G3)]
        self.masks = [(M / 255.0).astype(int) for M in (M0, M1, M2, M3)]
        return np.stack((G0, G1, G2, G3), axis=2), np.stack((M0, M1, M2, M3), axis=2)

    def make_white_balance(self):
        self.image = make_white_balance(self.image)
