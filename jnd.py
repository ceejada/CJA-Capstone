import numpy as np
import cv2
import random
import itertools
from itertools import combinations
import time
import csv
import os
from cja_utils import show_full_frame
from cja_utils import blend_lut
from cja_utils import sort_corners
from cja_utils import reverse_gogo

# Psychophysical testing

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
L_black_wall = 0.18
L_white_patch = 120
L_black_patch = 0.07
gamma = 2.6

gogo = np.arange(256, dtype=np.float32)
gogo = gogo / 255.0
gogo = np.power(gogo, gamma)

light_wall = (gogo * (L_white_wall - L_black_wall)) + L_black_wall
light_patch = (gogo * (L_white_patch - L_black_patch)) + L_black_patch

target = np.minimum(light_wall, L_white_patch)
patch0 = (gogo * (L_white_patch - L_black_patch)) + L_black_patch

log_file = "jnd.csv"
file_exists = os.path.isfile(log_file)

with open(log_file, "a", newline="") as f:
    writer = csv.writer(f)
    if not file_exists:
        writer.writerow([
            "name", "trial","image",
            "left_percent","right_percent",
            "response"
        ])

trial_num = 0
blocks = [0, 40, 80]

def select_blocks(block_list, step_size=5, max_difference=50):
    """
    Fixed base blocks:
    0, 40, 80

    Show every variation up to +50:

    0  -> 5, 10, 15, ..., 50
    40 -> 45, 50, 55, ..., 90
    80 -> 85, 90, 95, 100
         (capped at 100)
    """

    pairs = []

    for b in block_list:
        max_target = min(b + max_difference, 100)

        for target in range(b + step_size, max_target + 1, step_size):
            pairs.append((b, target))

    return pairs    
# create trial pairs once
trial_pairs = select_blocks(blocks)
random.shuffle(trial_pairs)

#highlights
im1 = cv2.imread("landscape.jpg")
im2 = cv2.imread("pier.jpg")

# shadows
im3 = cv2.imread("trees.jpg")
im4 = cv2.imread("city.jpg")

# neutrals
im5 = cv2.imread("ball.jpg")
im6 = cv2.imread("cam.jpg")
img_names = [
    "statue",
    "pier",
    "trees",
    "city",
    "ball",
    "cam"
]

imgs = [im1, im2, im3, im4, im5, im6]

# randomize order of trials
name = str(input("Enter your name: "))


trials = list(zip(imgs, img_names))

# randomly choose only 3 images
trials = random.sample(trials, 3)

# optional: shuffle their order
random.shuffle(trials)

for src, src_title in trials:

    trial_pairs = select_blocks(blocks)
    random.shuffle(trial_pairs)

    for left_percent, right_percent in trial_pairs:
        if random.choice([True, False]):
            left_percent, right_percent = right_percent, left_percent

        scale = min(half_w / src.shape[1], h / src.shape[0])
        new_w = int(src.shape[1] * scale)
        new_h = int(src.shape[0] * scale)

        src_resize = cv2.resize(
            src,
            (new_w, new_h),
            interpolation=cv2.INTER_CUBIC
        )

        left_img = np.zeros((h, w, 3), dtype=np.uint8)
        right_img = np.zeros((h, w, 3), dtype=np.uint8)

        y_offset = (h - new_h) // 2

        # center in left half
        x_left = (half_w - new_w) // 2
        left_img[
            y_offset:y_offset + new_h,
            x_left:x_left + new_w
        ] = src_resize

        # center in right half
        x_right = half_w + (half_w - new_w) // 2
        right_img[
            y_offset:y_offset + new_h,
            x_right:x_right + new_w
        ] = src_resize

        # -----------------------------
        # LEFT SIDE
        # -----------------------------
        wall_lut_percent = blend_lut(light_wall, target, left_percent)
        patch_lut_percent = blend_lut(patch0, target, left_percent)

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

        left_wall = cv2.LUT(left_img, wall_clip)
        left_patch = cv2.LUT(left_img, patch_clip)

        left_final = (
            left_patch * mask_left +
            left_wall * (~mask_left)
        )

        # -----------------------------
        # RIGHT SIDE
        # -----------------------------
        wall_lut_percent = blend_lut(light_wall, target, right_percent)
        patch_lut_percent = blend_lut(patch0, target, right_percent)

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

        right_wall = cv2.LUT(right_img, wall_clip)
        right_patch = cv2.LUT(right_img, patch_clip)

        right_final = (
            right_patch * mask_right +
            right_wall * (~mask_right)
        )

        # final!
        test = left_final + right_final
        test = np.clip(test, 0, 255).astype(np.uint8)

        show_full_frame(test)

        while True:
            key = cv2.waitKeyEx(0)
            if key == 2424832:
                response = "left"
                break
            elif key == 2555904:
                response = "right"
                break
            elif key == 27:
                cv2.destroyAllWindows()
                exit()

        with open(log_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                name,
                trial_num,
                src_title,
                left_percent,
                right_percent,
                response
            ])
        gray_screen = np.full((h, w, 3), 100, dtype=np.uint8)

        show_full_frame(gray_screen)
        cv2.waitKey(500)
        trial_num += 1


cv2.destroyAllWindows()
# show each percentage next to each other multiple times