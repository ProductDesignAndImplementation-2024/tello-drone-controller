import cv2
import os
from image_prosessing import preprocess_grid_image

def save_processed_image(image, filename="processed.png"):
    cv2.imwrite(filename, image)

def display_picture():
    files = os.listdir("pics")
    image_files = [file for file in files if file.startswith("picture_") and file.endswith(".png")]

    if not image_files:
        print("No pictures found.")
        return None, None

    numbers = [int(file.split('_')[1].split('.')[0]) for file in image_files]
    latest_number = max(numbers)
    latest_filename = f"pics/picture_{latest_number:03}.png"

    original_image = cv2.imread(latest_filename, cv2.IMREAD_GRAYSCALE)
    if original_image is None:
        print(f"Failed to load {latest_filename}.")
        return None, None

    processed_image = preprocess_grid_image(original_image)
    return original_image, processed_image
