#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Configuration and shared variables for Gesture PC Controller
"""

import cv2
import mediapipe as mp
import numpy as np
from pynput.keyboard import Controller as KeyboardController, Key, KeyCode
from pynput.mouse import Controller as MouseController

# Initialize MediaPipe hands module
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Initialize controllers
keyboard = KeyboardController()
mouse = MouseController()

# MediaPipe hand landmark indices for easy reference
THUMB_TIP = 4
INDEX_FINGER_TIP = 8
INDEX_FINGER_PIP = 6  # Proximal Interphalangeal Joint (middle joint)
MIDDLE_FINGER_TIP = 12
MIDDLE_FINGER_PIP = 10
RING_FINGER_TIP = 16
RING_FINGER_PIP = 14
PINKY_TIP = 20
PINKY_PIP = 18
WRIST = 0

# Gesture timing configuration
GESTURE_CONFIRMATION_TIME = 0.35  # Seconds to confirm gesture before tracking (increased from 0.1)
GESTURE_COOLDOWN_TIME = 0.15  # Seconds between gesture detections
MOVEMENT_THRESHOLD = 30  # Minimum X movement to trigger a keystroke
COOLDOWN_FRAMES = 10  # Wait this many frames after a key press