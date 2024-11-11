import cv2
import numpy as np

def nothing(x):
    pass

def save_processed_image():
    image = display_picture()
    cv2.imwrite("processed.png", image)

cv2.namedWindow("Before and After")
cv2.createTrackbar("Threshold", "Before and After", 25, 255, nothing)

min_area = 64
max_area = 10000

cv2.namedWindow("Landing Pad Detection")
#cv2.resizeWindow("Landing Pad Detection", 640, 400)
cv2.createTrackbar("Min Area", "Landing Pad Detection", min_area, 10000, nothing)
cv2.createTrackbar("Max Area", "Landing Pad Detection", max_area, 10000, nothing)

def display_picture():
    original_image = cv2.imread("picture.png", cv2.IMREAD_GRAYSCALE)
    #print(original_image.shape)
    # remove excess pixels from image
    #original_image = cv2.resize(original_image, [320, 240])

    threshold_value = cv2.getTrackbarPos("Threshold", "Before and After")

    _, processed_image = cv2.threshold(original_image, threshold_value, 255, cv2.THRESH_BINARY_INV)

    combined_image = np.hstack((original_image, processed_image))
    cv2.imshow("Before and After", combined_image)

    return processed_image

processed_image = display_picture()

def find_landing_pad(processed_image):
    min_area = cv2.getTrackbarPos("Min Area", "Landing Pad Detection")
    max_area = cv2.getTrackbarPos("Max Area", "Landing Pad Detection")

    processed_image_2 = processed_image.copy()
    processed_image_2 = cv2.cvtColor(processed_image_2, cv2.COLOR_GRAY2BGR)
    contours, _ = cv2.findContours(processed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    landing_pad_contour = None
    landingpad_xy = (0, 0)

    for contour in contours:    
        area = cv2.contourArea(contour)
        if min_area < area < max_area:
            landing_pad_contour = contour
            break

    if landing_pad_contour is not None:
        x, y, w, h = cv2.boundingRect(landing_pad_contour)
        rect_color = (0, 0, 255)
        cv2.rectangle(processed_image_2, (x, y), (x + w, y + h), rect_color, 2)

        landingpad_xy = ((x + (w / 2), y + (h / 2)))
    
    # Show the result
    cv2.imshow("Landing Pad Detection", processed_image_2)
    return landingpad_xy

if __name__ == "__main__":
    while True:
        # Exit on pressing 'x'

        key = cv2.waitKey(1)
        if key &0xFF == ord('x'):
            break
        elif key &0xFF == ord('p'):
            find_landing_pad(processed_image)
            
    cv2.waitKey(0)
    cv2.destroyAllWindows()
