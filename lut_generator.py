import matplotlib.pyplot as plt
import numpy as np
import cv2
from cja_utils import show_full_frame
from cja_utils import blend_lut
from cja_utils import sort_corners
from cja_utils import reverse_gogo

def build_soft_roll(lut, white_limit, percent):
    lut = np.copy(lut)

    p = percent / 100.0

    frac = 0.85

    # move start based on percent
    start_full = frac * white_limit
    start = white_limit - (white_limit - start_full) * p

    # keep end tied to start for stability
    end = white_limit + (white_limit - start)

    m = (white_limit - start) / ((end - start) ** 0.75)

    mask = (lut >= start) & (lut <= end)
    lut[mask] = start + m * (lut[mask] - start) ** 0.75

    lut = np.minimum(lut, white_limit)
    return lut

from screen_process import (
    get_masks
)

mask_patch, mask_wall, h, w = get_masks(
    "march28.jpg"
)

half_w = w // 2
mask_left = np.zeros_like(mask_patch)
mask_right = np.zeros_like(mask_patch)

mask_left[:, :half_w] = mask_patch[:, :half_w]
mask_right[:, half_w:] = mask_patch[:, half_w:]

L_white_wall = 350
L_black_wall = 1
L_white_patch = 100
L_black_patch = 0.05
gamma = 2.6

gogo = np.arange(256, dtype=np.float32)
gogo = gogo / 255.0
gogo = np.power(gogo, gamma)

def mrow(percent):
    light_wall = (gogo * (L_white_wall - L_black_wall)) + L_black_wall
    light_patch = (gogo * (L_white_patch - L_black_patch)) + L_black_patch

    #target = build_soft_roll(light_wall, L_white_patch, percent)
    target = np.minimum(light_wall, L_white_patch)
    patch0 = (gogo * (L_white_wall - L_black_patch)) + L_black_patch

    wall_lut_percent = blend_lut(light_patch, target, percent)
    patch_lut_percent = blend_lut(patch0, target, percent)

    wall_cv = reverse_gogo(
        wall_lut_percent,
        L_black_wall,
        L_white_wall,
        gamma
    )

    patch_cv = reverse_gogo(
        patch_lut_percent,
        L_black_patch,
        L_white_patch,
        gamma
    )

    wall_clip = np.clip(wall_cv, 0, 255).astype(np.uint8)
    patch_clip = np.clip(patch_cv, 0, 255).astype(np.uint8)
    return wall_clip, patch_clip

patch0 = (gogo * (L_white_wall - L_black_patch)) + L_black_patch
patch = reverse_gogo(patch0, L_black_patch, L_white_patch, gamma)
clip = np.clip(patch, 0, 255).astype(np.uint8)

#wall_clip0, patch_clip0 = mrow(0)
wall_clip25, patch_clip25 = mrow(25)
wall_clip50, patch_clip50 = mrow(50)
wall_clip75, patch_clip75 = mrow(75)
wall_clip100, patch_clip100 = mrow(100)

x = np.arange(256, dtype=np.uint8)

plt.figure(figsize=(10, 6))
#plt.plot(x, x, label="1 to 1")
plt.plot(x, x, label="Patch 0%")
plt.plot(x, patch_clip25, label="Patch 25%")
plt.plot(x, patch_clip50, label="Patch 50%")
plt.plot(x, patch_clip75, label="Patch 75%")
plt.plot(x, patch_clip100, label="Patch 100%")

#plt.plot(x, x, label="Patch Clip 0%")

plt.title("Patch LUTs")
plt.legend()
plt.grid(True)
plt.show()
