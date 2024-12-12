from djitellopy import Tello
import cv2
import time
import math
import numpy as np
import image_processing as imgp
import image_utils as img_utils
import intersection_finder as int_finder
import json

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

def move_over_triangle(coords, offset = 25):
    x, y = coords
    ret_x = 0
    ret_y = 0
    if x < 160 - offset:
        ret_x = -1
    elif x > 160 + offset:
        ret_x = 1
    
    if y < 120 - offset:
        ret_y = -1
    elif y > 120 + offset:
        ret_y = 1
    return (ret_x, ret_y)

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

def get_direction_vector(a, b, normalized = True):
    dir_x = a[0] - b[0]
    dir_y = a[1] - b[1]
    if normalized == False:
        return dir_x, dir_y

    magnitude = math.sqrt(dir_x**2 + dir_y**2)
    if magnitude == 0:
        return 0, 0

    return dir_x / magnitude, dir_y / magnitude

def calculate_angle_between_dir_vectors(v1, v2):
    dot_product = v1[0] * v2[0] + v1[1] * v2[1]
    dot_product = max(min(dot_product, 1.0), -1.0)
    angle_radians = math.acos(dot_product)
    
    cross_product = v1[0] * v2[1] - v1[1] * v2[0]
    if cross_product < 0:
        angle_radians = -angle_radians

    return angle_radians

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

                dir_x, dir_y = get_direction_vector(right_angle_point, end_point, True)

                # bottom right == (-0.7071067811865475, -0.7071067811865475)
                angle = calculate_angle_between_dir_vectors((-0.7071067811865475, -0.7071067811865475), (dir_x, dir_y))
                angle_degrees = math.degrees(angle)

                return angle_degrees, right_angle_point
    # no triangle found
    return None, None

def take_picture(tello: Tello):
    #latest_number = img_utils.get_latest_picture_number();
    #next_number = latest_number  +1 if latest_number != None else 0
    #filename = f"pics/picture_{next_number:03}.png"
    filename = "picture.png"

    # Take and save the picture
    frame_read = tello.get_frame_read()
    cv2.imwrite(filename, frame_read.frame)

    # Process and save the cropped version
    original_image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    cropped_image = original_image[0:240, 0:320]
    cv2.imwrite(filename, cropped_image)

def try_get_triangle_angle(tello: Tello):
    take_picture(tello)
    imgp.save_processed_image()
    image = cv2.imread('processed.png')

    angle, loc = detect_triangle_shape(image)
    for i in range(0, 5):
        if angle != None:
            break

        if i % 2:
            tello.move_down(20)
        else:
            tello.move_up(20)

        take_picture(tello)
        imgp.save_processed_image()
        image = cv2.imread('processed.png')

        angle, loc = detect_triangle_shape(image)

    return angle

def try_get_triangle_loc(tello: Tello):
    take_picture(tello)
    imgp.save_processed_image()
    image = cv2.imread('processed.png')

    angle, loc = detect_triangle_shape(image)
    for i in range(0, 5):
        if loc != None:
            break

        if i % 2:
            tello.move_down(20)
        else:
            tello.move_up(20)

        take_picture(tello)
        imgp.save_processed_image()
        image = cv2.imread('processed.png')

        angle, loc = detect_triangle_shape(image)

    return loc
        

def drone_autopilot_takeoff(tello: Tello):
    tello.takeoff()
    tello.move_up(30) # test for correct amount

    auto_align = align_drone_correctly(tello)
    if auto_align == False:
        # manual control required
        cv2.waitKey(0)

    # test for correct direction + amount
    tello.move_right(100)

def drone_autopilot_take_pictures():
    # take grid pictures
    # check if valid & intersections found
    # try find grid center
    # enable manual control if required
    return

def align_drone_correctly(tello: Tello):
    for i in range(0, 8):
        loc = try_get_triangle_loc(tello)
        x, y = move_over_triangle(loc, 30)

        amount = 20

        if x == 1:
            tello.move_back(amount)
        elif x == -1:
            tello.move_forward(amount)
        elif y == 1:
            tello.move_left(amount)
        elif y == -1:
            tello.move_right(amount)
        else:
            break

        #key = cv2.waitKey(0)
        #if key & 0xFF == ord('x'):
        #    tello.land()
        #    return False

    for i in range(0, 8):
        angle = try_get_triangle_angle(tello)
        if angle == None:
            return False # no triangle found (manual control required?)
        
        if 2 > angle > -2: # angle small enough
            return True
        
        if angle > 0:
            tello.rotate_clockwise(int(abs(angle)))
        else:
            tello.rotate_counter_clockwise(int(abs(angle)))

        if 2 > angle > -2:
            return True

    return False # failed to align => manual control

def autopilot(tello: Tello):
    #tello.takeoff()
    #tello.streamon()

    rotate = align_drone_correctly(tello)
    print(rotate)
    if rotate == False:
        return None

    tello.move_up(50)
    #key2 = cv2.waitKey(0)
    #if key2 & 0xFF == ord('x'):
    #    tello.land()
    #    return None
    tello.move_left(70)
    take_picture(tello)
    time.sleep(0.1)
    processed_image = imgp.display_picture()
    cv2.imwrite("processed.png", processed_image)
    cv2.imwrite("grid.png", processed_image)

    '''
    result_image = int_finder.find_path(False,True)
    cv2.imshow("Intersections and Paths", result_image)
    cv2.imwrite("intersections_and_paths.png", result_image)
    '''

    path = int_finder.find_path(False, False)
    # check if valid path

    tello.move_right(70)

    for i in range(0, 8):
        loc = try_get_triangle_loc(tello)
        x, y = move_over_triangle(loc, 20)

        amount = 20

        if x == 1:
            tello.move_back(amount)
        elif x == -1:
            tello.move_forward(amount)
        elif y == 1:
            tello.move_left(amount)
        elif y == -1:
            tello.move_right(amount)
        else:
            break

        #key = cv2.waitKey(0)
        #if key & 0xFF == ord('x'):
        #    tello.land()
        #    return False

    json_str = json.dumps(path)
    return json_str

if __name__ == "__main__":
    #image = cv2.imread('align_test.png')
    image = cv2.imread('processed.png')

    angle = detect_triangle_shape(image)
    print(angle)

    cv2.imshow('shapes', image)
    cv2.waitKey(0)

    cv2.destroyAllWindows()
