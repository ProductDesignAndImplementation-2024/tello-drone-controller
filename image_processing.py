import cv2
import numpy as np

# Load the original image in grayscale
original_image = cv2.imread("data/image.png", cv2.IMREAD_GRAYSCALE)

# Callback function for trackbar (does nothing but needed for OpenCV trackbars)
def nothing(x):
    pass

# Create a named window
cv2.namedWindow("Before and After")

# Create a trackbar for adjusting the threshold
cv2.createTrackbar("Threshold", "Before and After", 170, 255, nothing)

while True:
    # Get the current position of the trackbar
    threshold_value = cv2.getTrackbarPos("Threshold", "Before and After")

    # Apply the threshold with the current trackbar value
    _, processed_image = cv2.threshold(original_image, threshold_value, 255, cv2.THRESH_BINARY_INV)

    # Combine the original and processed images side by side
    combined_image = np.hstack((original_image, processed_image))

    # Display the combined image
    cv2.imshow("Before and After", combined_image)

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Close all OpenCV windows
cv2.destroyAllWindows()
