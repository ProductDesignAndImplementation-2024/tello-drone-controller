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

def drone_autopilot():
    tello = Tello()
    tello.connect()
    tello.streamon()

    tello.set_video_direction(tello.CAMERA_DOWNWARD)

    tello.takeoff()
    path = autopilot.autopilot(tello)
    tello.land()
    print(path)
    tello.end()
    return path