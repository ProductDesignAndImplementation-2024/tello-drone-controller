from djitellopy import Tello
import cv2
import time
import math
import numpy as np
import image_processing as imgp

def rotate_if_needed(landingpad):
    x, y = landingpad
    if y < 100:
        return -1
    elif y > 140:
        return 1
    return 0
    
def move_if_needed(landingpad):
    x, y = landingpad
    if x < 140:
        return -1
    elif x > 180:
        return 1
    return 0

def calculate_angle(p1, p2, p3):
    """
    Calculate the angle at p2 formed by the lines p1->p2 and p2->p3 using the dot product.
    """
    v1 = (p1[0] - p2[0], p1[1] - p2[1])  # Vector from p2 to p1
    v2 = (p3[0] - p2[0], p3[1] - p2[1])  # Vector from p2 to p3

    dot_product = v1[0] * v2[0] + v1[1] * v2[1]  # Dot product
    mag_v1 = math.sqrt(v1[0]**2 + v1[1]**2)  # Magnitude of v1
    mag_v2 = math.sqrt(v2[0]**2 + v2[1]**2)  # Magnitude of v2

    if mag_v1 * mag_v2 == 0:  # Avoid division by zero
        return 0

    cos_angle = dot_product / (mag_v1 * mag_v2)
    angle = math.degrees(math.acos(cos_angle))  # Convert from radians to degrees
    return angle

def count_white_pixels_in_box(image, contour, max_white_percentage=15):
    """
    Count the percentage of white pixels inside the bounding box of a contour.
    Returns True if the percentage of white pixels is within the acceptable threshold.
    """
    # Get bounding box of the contour
    x, y, w, h = cv2.boundingRect(contour)
    
    # Extract the region of interest (ROI) from the image
    roi = image[y:y+h, x:x+w]
    
    # Convert ROI to grayscale and threshold it to count white pixels
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, binary_roi = cv2.threshold(gray_roi, 200, 255, cv2.THRESH_BINARY)
    
    # Count the number of white pixels (255)
    white_pixels = np.sum(binary_roi == 255)
    
    # Calculate the total area of the bounding box
    total_area = w * h
    
    # Calculate the white pixel percentage
    white_pixel_percentage = (white_pixels / total_area) * 100
    
    # Check if the white pixel percentage exceeds the threshold
    return white_pixel_percentage <= max_white_percentage

def calculate_direction_vector(right_angle_point, avg_point):
    # Calculate the direction vector components
    dx = avg_point[0] - right_angle_point[0]
    dy = avg_point[1] - right_angle_point[1]
    
    # Calculate the magnitude of the vector (optional)
    magnitude = math.sqrt(dx ** 2 + dy ** 2)
    
    # Normalize the vector if magnitude is greater than 0
    if magnitude > 0:
        dx /= magnitude
        dy /= magnitude
    
    return (dx, dy), magnitude

def detect_triangle_shape(image, tolerance = 20):
    # Step 1: Preprocess the image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)

    # Step 2: Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 

    for contour in contours:
        # adjust epsilon if triangle is not detected
        approx = cv2.approxPolyDP(contour, 0.04 * cv2.arcLength(contour, True), True) 
        
        M = cv2.moments(contour) 
        if M['m00'] != 0.0: 
            x = int(M['m10']/M['m00']) 
            y = int(M['m01']/M['m00'])

        if len(approx) == 3: 
            if not count_white_pixels_in_box(image, contour, max_white_percentage=30):
                continue  # Skip this contour if white pixel percentage is too high

            #cv2.putText(image, 'Triangle', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 0, 255), 2)
            
            points = [tuple(point[0]) for point in approx]
            for point in points:
                x = point[0]
                y = point[1]
                cv2.circle(image, (x, y), 5, (220, 0, 255), 2)

            angles = [
                calculate_angle(points[1], points[0], points[2]),
                calculate_angle(points[2], points[1], points[0]),
                calculate_angle(points[0], points[2], points[1])
            ]

            i_prev = 0
            i_next = 0
            right_angle = -1
            for i, angle in enumerate(angles):
                print(angle)
                if 90 - tolerance <= angle <= 90 + tolerance:
                    #cv2.circle(image, points[i], 15, (0, 0, 255), 4)

                    i_prev = i - 1
                    if i_prev < 0:
                        i_prev = 2
                    
                    i_next = i + 1
                    if i_next > 2:
                        i_next = 0

                    cv2.line(image, points[i], points[i_prev], (255, 0 , 0), 2)
                    cv2.line(image, points[i], points[i_next], (0, 255, 0), 2)

                    right_angle = i

            if right_angle != -1:
                avg_x = (points[i_prev][0] + points[i_next][0]) / 2
                avg_y = (points[i_prev][1] + points[i_next][1]) / 2
    
                #cv2.circle(image, (int(avg_x), int(avg_y)), 15, (0, 255, 255), 2)

                right_angle_point = points[right_angle]  # Right angle vertex
                avg_point = (avg_x, avg_y)  # Average point
                direction_vector, magnitude = calculate_direction_vector(right_angle_point, avg_point)

                # Draw the direction vector (optional: scale it for visibility)
                scale = 64  # Scale factor to make the vector more visible
                end_point = (int(right_angle_point[0] + direction_vector[0] * scale),
                             int(right_angle_point[1] + direction_vector[1] * scale))
                cv2.arrowedLine(image, right_angle_point, end_point, (0, 255, 255), 2, tipLength=0.05)

if __name__ == "__main__":
    #image = cv2.imread('align_test.png')
    image = cv2.imread('processed.png')

    detect_triangle_shape(image)

    cv2.imshow('shapes', image)
    cv2.waitKey(0)

    cv2.destroyAllWindows()