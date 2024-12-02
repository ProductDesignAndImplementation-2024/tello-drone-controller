import cv2
import numpy as np
import math

def preprocess_grid_image(input_image, mask_strength=0.15):

    # Stacked blurred (reference vignette)
    stacked_blur_image = stacked_blur(input_image, kernel_size=15, iterations=10)


    # Invert
    correction_mask = 255 - stacked_blur_image


    # Correction mask
    brightened_image = input_image + (mask_strength * correction_mask).astype(np.uint8)
    darkened_image = input_image - (mask_strength * correction_mask).astype(np.uint8)


    # Brighten dark areas and darken bright areas
    corrected_image = np.where(input_image < 128, brightened_image, darkened_image)
    corrected_image = np.clip(corrected_image, 0, 255).astype(np.uint8)
    cv2.imshow("stackblur1", corrected_image)
    # Thresholding
    max_luminance = 70
    replace_value = 255

    mask = corrected_image <= max_luminance
    processed_frame = np.zeros_like(corrected_image)
    processed_frame[mask] = replace_value


    # Begone, small white areas
    min_size = 500
    processed_frame = remove_small_white_masses(processed_frame, min_size)

    return processed_frame


def stacked_blur(input_image, kernel_size=15, iterations=10):
    blurred_image = input_image.copy()
    for _ in range(iterations):
        blurred_image = cv2.blur(blurred_image, (kernel_size, kernel_size))
    cv2.imshow("stackblur2", blurred_image)
    return blurred_image


def remove_small_white_masses(binary_image, min_size):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_image, connectivity=8)
    filtered_image = np.zeros_like(binary_image)

    for label in range(1, num_labels):  # Skip the background
        area = stats[label, cv2.CC_STAT_AREA]
        if area >= min_size:
            filtered_image[labels == label] = 255

    return filtered_image
