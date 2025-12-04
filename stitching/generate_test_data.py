import numpy as np
import struct

def generate_test_binary():
    print("Generating test surround_view.binary for 1920x1080 split screen...")
    
    # Output resolution
    SW, SH = 1920, 1080
    
    # Input resolution (Camera)
    IW, IH = 1920, 1080
    
    # We want to create a 4-way split screen:
    # Top-Left: Front (Cam 0)
    # Top-Right: Right (Cam 3) - let's follow standard order Front, Left, Back, Right
    # But for split screen: 
    # TL: Front (0)
    # TR: Left (1)
    # BL: Back (2)
    # BR: Right (3)
    
    # Create the meshgrid for the output image
    xx, yy = np.meshgrid(np.arange(SW), np.arange(SH))
    
    # Initialize the table with 10 channels
    # Channels: [flag, weight, x0, y0, x1, y1, x2, y2, x3, y3]
    table = np.zeros((SH, SW, 10), dtype=np.float32)
    
    # Define quadrants
    # 0: TL, 1: TR, 2: BL, 3: BR
    mask_tl = (xx < SW/2) & (yy < SH/2)
    mask_tr = (xx >= SW/2) & (yy < SH/2)
    mask_bl = (xx < SW/2) & (yy >= SH/2)
    mask_br = (xx >= SW/2) & (yy >= SH/2)
    
    # --- Quadrant 0: Front (Cam 0) ---
    # Map TL output (0..SW/2, 0..SH/2) to Input (0..IW, 0..IH)
    # Scale factors
    scale_x = IW / (SW/2)
    scale_y = IH / (SH/2)
    
    table[mask_tl, 0] = 0 # Flag = 0 (Cam 0)
    table[mask_tl, 1] = 1.0 # Weight = 1.0
    # Map coordinates
    table[mask_tl, 2] = xx[mask_tl] * scale_x # x0
    table[mask_tl, 3] = yy[mask_tl] * scale_y # y0
    
    # --- Quadrant 1: Left (Cam 1) ---
    table[mask_tr, 0] = 1 # Flag = 1
    table[mask_tr, 1] = 1.0
    # Map TR output to Input
    # xx goes from SW/2 to SW. We need to shift it back to 0..SW/2 then scale
    table[mask_tr, 4] = (xx[mask_tr] - SW/2) * scale_x # x1
    table[mask_tr, 5] = yy[mask_tr] * scale_y # y1
    
    # --- Quadrant 2: Back (Cam 2) ---
    table[mask_bl, 0] = 2 # Flag = 2
    table[mask_bl, 1] = 1.0
    table[mask_bl, 6] = xx[mask_bl] * scale_x # x2
    table[mask_bl, 7] = (yy[mask_bl] - SH/2) * scale_y # y2
    
    # --- Quadrant 3: Right (Cam 3) ---
    table[mask_br, 0] = 3 # Flag = 3
    table[mask_br, 1] = 1.0
    table[mask_br, 8] = (xx[mask_br] - SW/2) * scale_x # x3
    table[mask_br, 9] = (yy[mask_br] - SH/2) * scale_y # y3
    
    # Save to binary
    table.tofile("surround_view.binary")
    print(f"Generated surround_view.binary ({SW}x{SH}x10)")

if __name__ == "__main__":
    generate_test_binary()
