from djitellopy import Tello
import cv2
import time
import numpy as np
import os
import image_processing as imgp
import drone_autopilot as autopilot
import drone_manual_control as manual_control
import image_utils as img_utils
import intersection_finder as int_finder

tello = Tello()
tello.connect()
tello.streamon()

tello.set_video_direction(tello.CAMERA_DOWNWARD)

cv2.namedWindow("Tello-drone Info", cv2.WINDOW_NORMAL)

def fetch_tello_info_window_data():
    info_battery = tello.get_battery()
    info_battery_estimate = 780 - (int)((90 - (info_battery - 10)) * 8.66)
    if info_battery_estimate < 0:
        info_battery_estimate = 0

    info_flighttime = tello.get_flight_time()
    info_height = tello.get_height()

    info_speed_x = tello.get_speed_x()
    info_speed_y = tello.get_speed_y()
    info_speed_z = tello.get_speed_z()

    info_roll = tello.get_roll()
    info_pitch = tello.get_pitch()

    info_temp_avg = tello.get_temperature()
    info_temp_min = tello.get_lowest_temperature()
    info_temp_max = tello.get_highest_temperature()

    return {
        "Battery": f"{info_battery}",
        "Battery Estimate": f"{info_battery_estimate}",
        "Flight time": f"{info_flighttime}s",
        "Height": f"{info_height}cm",
        
        "Speed": f"{info_speed_x}, {info_speed_y}, {info_speed_z}",
        "Roll-Pitch": f"{info_roll}, {info_pitch}",
        "Temp (avg, min, max)": f"{info_temp_avg}C, {info_temp_min}C, {info_temp_max}C"
    }

def display_tello_info_window(info):
    info_image = np.zeros((480, 600, 3), dtype=np.uint8)

    font = cv2.FONT_HERSHEY_COMPLEX
    font_scale = 0.6
    color = (255, 255, 255)
    line_type = 2

    y_offset = 15
    for key, value in info.items():
        text = f"{key}: {value}"

        # battery text color
        if key == "Battery" or key == "Battery Estimate":
            if int(value) < 15:
                color = (0, 0, 255)
            elif int(value) < 30:
                color = (0, 255, 255)
            else:
                color = (0, 255, 0)
        else:
            color = (255, 255, 255)
                
        cv2.putText(info_image, text, (10, y_offset), font, font_scale, color, line_type)
        y_offset += 45

    cv2.imshow("Tello-drone Info", info_image)

def take_picture():
    latest_number = img_utils.get_latest_picture_number();
    next_number = latest_number  +1 if latest_number != None else 0
    filename = f"pics/picture_{next_number:03}.png"

    # Take and save the picture
    frame_read = tello.get_frame_read()
    cv2.imwrite(filename, frame_read.frame)

    # Process and save the cropped version
    original_image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    cropped_image = original_image[0:240, 0:320]
    cv2.imwrite(filename, cropped_image)

autopilot_is_enabled = False
landingpad_old_loc = (-1, -1)

def run_autopilot(landingpad_xy):
    x, y = landingpad_xy
    if x == -1:
        print("Autopilot enabled, landingpad not found!")
        return False

    rotation = autopilot.rotate_if_needed(landingpad_xy)
    move = autopilot.move_if_needed(landingpad_xy)
    print(rotation)
    if rotation == 1:
        tello.rotate_clockwise(20 * rotation)
    elif rotation == -1:
        tello.rotate_counter_clockwise(20 * -rotation)
    else:
        if move == 1:
            tello.move_back(20 * move)
        elif move == -1:
            tello.move_forward(20 * -move)
        else:
            tello.land()
            return False

    return True

def take_grid_picture():
    take_picture()

    # rotate picture
    # align grid if needed
    # intersection_finder.py => if ok return to landingpad
    # => if not ok, realign + new picture

def drone_get_path():
    return autopilot.autopilot(tello)

while True:
    tello_info = fetch_tello_info_window_data()
    display_tello_info_window(tello_info)

    processed = imgp.display_picture()
    landingpad_xy = imgp.find_landing_pad(processed, landingpad_old_loc)
    landingpad_old_loc = landingpad_xy

    key = cv2.waitKey(1)
    if key & 0xFF == ord('x'):
        break
    elif key & 0xFF == ord('t'):
        #tello.streamon()
        tello.takeoff()
    elif key & 0xFF == ord('l'):
        landingpad_old_loc = (-1, -1)
        tello.land()
        #tello.streamoff()
    elif key & 0xFF == ord('p'):
        take_picture()
    elif key & 0xFF == ord('m'):
        rotate = autopilot.align_drone_correctly(tello)
        print(rotate)
        if rotate == False:
            continue

        tello.move_up(70)
        key2 = cv2.waitKey(0)
        if key2 & 0xFF == ord('x'):
            tello.land()
            break
        tello.move_right(85)
        take_picture()
        time.sleep(0.1)
        processed_image = imgp.display_picture()
        cv2.imwrite("processed.png", processed_image)

        result_image = int_finder.find_path(False,True)
        #if result_image != None:
        cv2.imshow("Intersections and Paths", result_image)
        cv2.imwrite("intersections_and_paths.png", result_image)
        #print(path)

        tello.move_left(85)
    elif key & 0xFF == ord('o'):
        take_picture()
        time.sleep(0.1)

        processed = imgp.display_picture()
        landingpad_xy = imgp.find_landing_pad(processed, landingpad_old_loc)
        landingpad_old_loc = landingpad_xy

        autopilot_is_enabled = run_autopilot(landingpad_xy)
        if autopilot_is_enabled == False:
            landingpad_old_loc = (-1, -1)
    elif key & 0xFF == ord('b'):
        if autopilot_is_enabled:
            autopilot_is_enabled = False
        else:
            autopilot_is_enabled = True
            if autopilot_is_enabled == False:
                landingpad_old_loc = (-1, -1)
    elif manual_control.move_if_required(tello, key):
        continue
    else:
        if autopilot_is_enabled:
            take_picture()
            time.sleep(0.1)

            processed = imgp.display_picture()
            landingpad_xy = imgp.find_landing_pad(processed, landingpad_old_loc)
            landingpad_old_loc = landingpad_xy

            autopilot_is_enabled = run_autopilot(landingpad_xy)
            if autopilot_is_enabled == False:
                landingpad_old_loc = (-1, -1)
        
cv2.destroyAllWindows()
tello.end()
