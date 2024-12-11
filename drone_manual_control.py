from djitellopy import Tello
import numpy as np
import os

# WASD = moveement
# E-Q = rotate
# R-F = up-down
# K = flip forward
def move_if_required(tello: Tello, key: int):
    actions = {
        ord('w'): lambda: tello.move_forward(20),
        ord('s'): lambda: tello.move_back(20),
        ord('a'): lambda: tello.move_left(20),
        ord('d'): lambda: tello.move_right(20),
        ord('e'): lambda: tello.rotate_clockwise(3),
        ord('q'): lambda: tello.rotate_counter_clockwise(3),
        ord('E'): lambda: tello.rotate_clockwise(20),
        ord('Q'): lambda: tello.rotate_counter_clockwise(20),
        ord('r'): lambda: tello.move_up(20),
        ord('f'): lambda: tello.move_down(20),
        ord('k'): lambda: tello.flip_forward,
    }

    action = actions.get(key & 0xFF)
    if action:
        action()
        return True

    return False
