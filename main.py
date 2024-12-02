import cv2
from image_io import display_picture, save_processed_image
from gui import show_combined_image

if __name__ == "__main__":
    window_name = "Before and After"
    cv2.namedWindow(window_name)

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('x'):  # Exit on 'x'
            break
        elif key == ord('p'):  # Process and save on 'p'
            original_image, processed_image = display_picture()
            if processed_image is not None:
                show_combined_image(original_image, processed_image, window_name)
                save_processed_image(processed_image)

    cv2.destroyAllWindows()
