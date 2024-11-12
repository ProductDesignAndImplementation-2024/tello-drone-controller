import cv2
import numpy as np
import math

def nothing(x):
    pass

def save_processed_image():
    image = display_picture()
    cv2.imwrite("processed.png", image)

cv2.namedWindow("Before and After")
cv2.createTrackbar("Threshold", "Before and After", 25, 255, nothing)

min_area = 100
max_area = 800

#cv2.namedWindow("Landing Pad Detection")
cv2.createTrackbar("Min Area", "Before and After", min_area, 6000, nothing)
cv2.createTrackbar("Max Area", "Before and After", max_area, 6000, nothing)

def display_picture():
    original_image = cv2.imread("picture.png", cv2.IMREAD_GRAYSCALE)

    threshold_value = cv2.getTrackbarPos("Threshold", "Before and After")

    _, processed_image = cv2.threshold(original_image, threshold_value, 255, cv2.THRESH_BINARY_INV)

    combined_image = np.hstack((original_image, processed_image))
    #cv2.imshow("Before and After", combined_image)

    return processed_image

processed_image = display_picture()

def find_landing_pad(processed_image, landingpad_old_loc):
    min_area = cv2.getTrackbarPos("Min Area", "Before and After")
    max_area = cv2.getTrackbarPos("Max Area", "Before and After")

    processed_image_2 = processed_image.copy()
    processed_image_2 = cv2.cvtColor(processed_image_2, cv2.COLOR_GRAY2BGR)
    contours, _ = cv2.findContours(processed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    landing_pad_contour = None
    old_x, old_y = landingpad_old_loc
    landingpad_xy = (-1, -1)

    for contour in contours:    
        area = cv2.contourArea(contour)
        if min_area < area < max_area:
            if old_x == -1:
                landing_pad_contour = contour
                break

            x, y, w, h = cv2.boundingRect(landing_pad_contour)
            
            dist = pow((old_x + (x + (w / 2))), 2) + pow((old_y + (y + (h / 2))), 2)
            dist = math.sqrt(dist)
            if dist < 64:
                landing_pad_contour = contour
                break

    if landing_pad_contour is not None:
        x, y, w, h = cv2.boundingRect(landing_pad_contour)
        rect_color = (0, 0, 255)
        cv2.rectangle(processed_image_2, (x, y), (x + w, y + h), rect_color, 2)

        landingpad_xy = ((x + (w / 2), y + (h / 2)))
    
    # Show the result
    cv2.imshow("Before and After", processed_image_2)
    return landingpad_xy

if __name__ == "__main__":
    while True:
        # Exit on pressing 'x'

        key = cv2.waitKey(1)
        if key &0xFF == ord('x'):
            break
        elif key &0xFF == ord('p'):
            processed_image = display_picture()
            find_landing_pad(processed_image)
            
    cv2.waitKey(0)
    cv2.destroyAllWindows()
