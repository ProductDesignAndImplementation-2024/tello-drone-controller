import cv2
import numpy as np
import math
import os

def get_latest_picture_number():
    files = os.listdir("pics")
    image_files = [
        file for file in files if file.startswith("picture_") and file.endswith(".png")
    ]
    
    # If there are image files, get the latest one based on numeric suffix
    if image_files:
        # Extract numbers and find the latest image
        numbers = [
            int(file.split('_')[1].split('.')[0]) for file in image_files
        ]
        latest_number = max(numbers)
        return latest_number
    else:
        return None