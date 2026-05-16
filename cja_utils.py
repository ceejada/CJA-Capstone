import numpy as np
import cv2

def build_soft_roll(lut, white_limit):
    lut = np.copy(lut)

    frac = 0.85
    start = frac * white_limit
    end = white_limit / frac
    m = (white_limit - start) / ((end - start) ** 0.75)

    mask = (lut >= start) & (lut <= end)
    lut[mask] = start + m * (lut[mask] - start) ** 0.75


    lut = np.minimum(lut, white_limit)
    return lut

def blend_lut(unity_lut, corrected_lut, percent):
    """
    0%   = unity
    100% = full correction
    """
    t = np.clip(percent / 100.0, 0.0, 1.0)
    return (1 - t) * unity_lut + t * corrected_lut

def show_full_frame(frame):
    cv2.namedWindow('Full Screen', cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty('Full Screen', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow('Full Screen', frame)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

def sort_corners(pts):
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]

    return np.float32([tl, tr, br, bl])

def reverse_gogo(lut, L_min, L_max, gamma):
    norm = (lut - L_min) / (L_max - L_min)
    inv = np.power(norm, 1 / gamma)
    cv_out = inv * 255
    return cv_out
