#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Alt+F4 gesture module - DISABLED
This module has been disabled as requested.
"""

import time
import cv2
from gestures.base import GestureState

# Create state for Alt+F4 gesture (kept for compatibility)
alt_f4_state = GestureState()

def process_alt_f4_gesture(image, hand_landmarks, fps, image_width, image_height):
    """
    Alt+F4 gesture has been disabled.
    This function now simply returns the image without any processing.
    """
    # Simply return the image without any processing
    return image

def check_partial_hand(hand_landmarks):
    """
    This function has been disabled and now always returns False.
    """
    # Function disabled
    return False