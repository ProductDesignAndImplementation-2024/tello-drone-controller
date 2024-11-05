from djitellopy import Tello
import cv2
import time
import numpy as np

tello = Tello()
tello.connect()

tello.set_video_direction(tello.CAMERA_DOWNWARD)

tello.streamon()

def nothing(x):
    pass

cv2.namedWindow("Tello-drone Info", cv2.WINDOW_NORMAL)
cv2.namedWindow("Before and After")
cv2.createTrackbar("Threshold", "Before and After", 170, 255, nothing)

def fetch_tello_info_window_data():
    info_battery = tello.get_battery()
    info_flighttime = tello.get_flight_time()

    #time.sleep(1)

    return {
        "Battery": f"{info_battery}",
        "Flight time": f"{info_flighttime}"
    }

def display_tello_info_window(info):
    info_image = np.zeros((300, 400, 3), dtype=np.uint8)

    font = cv2.FONT_HERSHEY_COMPLEX
    font_scale = 0.6
    color = (255, 255, 255)
    line_type = 2

    y_offset = 30
    for key, value in info.items():
        text = f"{key}: {value}"

        # battery text color
        if key == "Battery":
            if int(value) < 15:
                color = (255, 0, 0)
            elif int(value) < 30:
                color = (255, 255, 0)
            else:
                color = (0, 255, 0)
        else:
            color = (255, 255, 255)
                
        cv2.putText(info_image, text, (10, y_offset), font, font_scale, color, line_type)
        y_offset += 30

    cv2.imshow("Tello-drone Info", info_image)

def take_picture():
    frame_read = tello.get_frame_read()
    time.sleep(0.5)
    cv2.imwrite("picture.png", frame_read.frame)

def display_picture():
    original_image = cv2.imread("picture.png", cv2.IMREAD_GRAYSCALE)
    threshold_value = cv2.getTrackbarPos("Threshold", "Before and After")

    _, processed_image = cv2.threshold(original_image, threshold_value, 255, cv2.THRESH_BINARY_INV)

    combined_image = np.hstack((original_image, processed_image))
    cv2.imshow("Before and After", combined_image)

while True:
    tello_info = fetch_tello_info_window_data()
    display_tello_info_window(tello_info)

    display_picture()

    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break
    elif key & 0xFF == ord('t'):
        tello.takeoff()
    elif key & 0xFF == ord('l'):
        tello.land()
    elif key & 0xFF == ord('p'):
        take_picture()
        
tello.land()

cv2.destroyAllWindows()
tello.end()
