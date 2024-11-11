from djitellopy import Tello
import cv2
import time
import numpy as np
import image_processing as imgp

def rotate_if_needed(landingpad):
    x, y = landingpad
    print(x)
    print(y)

    if x < 140:
        return -1
    elif x > 180:
        return 1
    return 0
    
def move_if_needed(landingpad):
    x, y = landingpad
    if y < 100:
        return -1
    elif y > 140:
        return 1
    return 0