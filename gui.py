import cv2
import numpy as np

def nothing(x):
    """Dummy callback"""
    pass

def show_combined_image(original_image, processed_image, window_name="Before and After"):
    combined_image = np.hstack((original_image, processed_image))
    cv2.imshow(window_name, combined_image)
