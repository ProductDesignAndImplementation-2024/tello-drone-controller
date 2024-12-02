import cv2
import numpy as np
from image_prosessing import preprocess_grid_image

def process_and_display_gif(gif_path):

    window_name = "GIF Processing"
    cv2.namedWindow(window_name)

    while True:  # Infinite loop for playback
        cap = cv2.VideoCapture(gif_path)
        if not cap.isOpened():
            print(f"Error: Unable to open GIF file {gif_path}")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            processed_frame = preprocess_grid_image(gray_frame)

            combined = np.hstack((gray_frame, processed_frame))

            cv2.imshow(window_name, combined)

            if cv2.waitKey(100) & 0xFF == ord('x'):
                cap.release()
                cv2.destroyAllWindows()
                return

        cap.release()

if __name__ == "__main__":
    input_gif = "input.gif"
    process_and_display_gif(input_gif)
