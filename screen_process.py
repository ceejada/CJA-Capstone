import cv2
import numpy as np
from cja_utils import sort_corners

# Image processing for capstone setup

def get_masks(
    image_path,
    img_w=1920,
    img_h=1080
):
    img = cv2.imread(image_path)

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

    blur = cv2.GaussianBlur(
        gray,
        (9, 9),
        0
    )

    _, thresh = cv2.threshold(
        blur,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    cnts, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    for cnt in cnts:
        area = cv2.contourArea(cnt)

        if area < 500:
            continue

        epsilon = 0.02 * cv2.arcLength(
            cnt,
            True
        )

        approx = cv2.approxPolyDP(
            cnt,
            epsilon,
            True
        )

        if len(approx) == 4:
            quad = approx.reshape(
                4, 2
            ).astype(np.float32)

            src_quad = sort_corners(quad)

            dst_quad = np.float32([
                [0, 0],
                [img_w, 0],
                [img_w, img_h],
                [0, img_h]
            ])

            M = cv2.getPerspectiveTransform(
                src_quad,
                dst_quad
            )

            warped = cv2.warpPerspective(
                img,
                M,
                (img_w, img_h)
            )

            gray_warped = cv2.cvtColor(
                warped,
                cv2.COLOR_BGR2GRAY
            )

            _, mask = cv2.threshold(
                gray_warped,
                200,
                255,
                cv2.THRESH_BINARY
            )

            inv_mask = (mask == 0)
            h, w = mask.shape[:2]
            srcmask = np.zeros((h, w), dtype=bool)
            srcmask[inv_mask] = True

            mask_patch = srcmask[:, :, np.newaxis]
            mask_wall = ~mask_patch

            return mask_patch, mask_wall, h, w

    raise Exception("No valid screen found")
