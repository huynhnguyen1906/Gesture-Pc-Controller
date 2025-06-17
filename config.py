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

# --- Gesture behavior configuration ---

# Minimum X movement to trigger a keystroke
DEFAULT_MOVEMENT_THRESHOLD = 20

# Minimum Y movement for Alt+Tab gesture
DEFAULT_Y_MOVEMENT_THRESHOLD = 20

# Seconds to confirm gesture before tracking
DEFAULT_GESTURE_CONFIRMATION_TIME = 0.5

# Seconds between gesture detections
DEFAULT_GESTURE_COOLDOWN_TIME = 0.5

# Wait this many frames after a key press
DEFAULT_COOLDOWN_FRAMES = 5

# --- Mouse control settings ---

# Smoothing factor for mouse movement (0-1, higher means smoother but slower)
DEFAULT_SMOOTHING_FACTOR = 0.7

# Increase sensitivity for mouse movement
DEFAULT_SENSITIVITY_MULTIPLIER = 1.1

# Default screen resolution, will be updated
DEFAULT_SCREEN_WIDTH = 1920
DEFAULT_SCREEN_HEIGHT = 1080