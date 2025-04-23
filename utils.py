#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Utility functions for Gesture PC Controller
"""

import cv2
import numpy as np
import mediapipe as mp

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

# Variables for tracking movement and gesture confirmation
prev_x = None
gesture_active = False
movement_threshold = 30  # Minimum X movement to trigger a keystroke
cooldown_counter = 0
cooldown_frames = 10  # Wait this many frames after a key press

gesture_confirmation_time = 0.1  # Seconds to confirm gesture before tracking
gesture_confirmation_counter = 0
gesture_cooldown_time = 0.15  # Seconds between gesture detections
gesture_cooldown_counter = 0
last_gesture_time = 0
confirmed_gesture = False

def calculate_distance(point1, point2):
    """Calculate Euclidean distance between two points"""
    return np.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2)

def draw_gesture_info(image, is_correct, is_active_hand=True):
    """Draw gesture information on the image"""
    # Draw background rectangle for better text visibility
    cv2.rectangle(image, (5, 5), (300, 140), (0, 0, 0), -1)
    cv2.rectangle(image, (5, 5), (300, 140), (255, 255, 255), 2)
    
    # Draw gesture status
    gesture_status = "Correct Gesture" if is_correct else "Wrong Gesture"
    gesture_color = (0, 255, 0) if is_correct else (0, 0, 255)
    
    cv2.putText(
        image,
        gesture_status,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        gesture_color,
        2
    )
    
    # Draw hand activity info
    hand_status = "Gesture Active" if is_active_hand else "Gesture Inactive"
    hand_color = (0, 255, 0) if is_active_hand else (255, 165, 0)  # Green if active, orange if inactive
    
    cv2.putText(
        image,
        hand_status,
        (10, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        hand_color,
        2
    )
    
    # Draw instruction
    cv2.putText(
        image,
        "2 fingers extended, 2 bent",
        (10, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1
    )
    
    return image

def enhance_visualization(image, hand_landmarks, mp_hands, mp_drawing):
    """Add enhanced visualization of hand landmarks"""
    # Draw basic hand landmarks
    mp_drawing.draw_landmarks(
        image,
        hand_landmarks,
        mp_hands.HAND_CONNECTIONS
    )
    
    # Highlight important finger tips with circles
    h, w, _ = image.shape
    landmarks = hand_landmarks.landmark
    
    # Draw circles on index and middle finger tips (in green)
    for idx in [INDEX_FINGER_TIP, MIDDLE_FINGER_TIP]:
        cx, cy = int(landmarks[idx].x * w), int(landmarks[idx].y * h)
        cv2.circle(image, (cx, cy), 12, (0, 255, 0), -1)
        cv2.circle(image, (cx, cy), 12, (255, 255, 255), 2)
    
    # Draw circles on ring and pinky finger tips (in red)
    for idx in [RING_FINGER_TIP, PINKY_TIP]:
        cx, cy = int(landmarks[idx].x * w), int(landmarks[idx].y * h)
        cv2.circle(image, (cx, cy), 8, (0, 0, 255), -1)
        cv2.circle(image, (cx, cy), 8, (255, 255, 255), 1)
    
    return image