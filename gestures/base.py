#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Base classes and functions for gesture detection
"""

import time
import cv2
import numpy as np
from config import (
    mp_hands,
    THUMB_TIP, INDEX_FINGER_TIP, INDEX_FINGER_PIP, MIDDLE_FINGER_TIP,
    MIDDLE_FINGER_PIP, RING_FINGER_TIP, RING_FINGER_PIP, PINKY_TIP, PINKY_PIP, WRIST,
    DEFAULT_MOVEMENT_THRESHOLD, DEFAULT_Y_MOVEMENT_THRESHOLD, DEFAULT_GESTURE_CONFIRMATION_TIME,
    DEFAULT_GESTURE_COOLDOWN_TIME, DEFAULT_COOLDOWN_FRAMES,
    DEFAULT_SMOOTHING_FACTOR, DEFAULT_SENSITIVITY_MULTIPLIER,
    DEFAULT_SCREEN_WIDTH, DEFAULT_SCREEN_HEIGHT
)

class GestureState:
    """Base class for tracking gesture state"""
    def __init__(self):
        self.prev_x = None
        self.prev_y = None
        self.gesture_active = False

        self.movement_threshold = DEFAULT_MOVEMENT_THRESHOLD
        self.cooldown_counter = 0
        self.cooldown_frames = DEFAULT_COOLDOWN_FRAMES

        self.gesture_confirmation_time = DEFAULT_GESTURE_CONFIRMATION_TIME
        self.gesture_confirmation_counter = 0
        self.gesture_cooldown_time = DEFAULT_GESTURE_COOLDOWN_TIME
        self.gesture_cooldown_counter = 0
        self.confirmed_gesture = False

        # For open-close hand gesture
        self.hand_open = False
        self.hand_closed = False
        self.open_hand_confirmed = False
        self.last_hand_state_change_time = 0        # For mouse control
        self.smoothing_factor = DEFAULT_SMOOTHING_FACTOR
        self.screen_width = DEFAULT_SCREEN_WIDTH
        self.screen_height = DEFAULT_SCREEN_HEIGHT
        self.sensitivity_multiplier = DEFAULT_SENSITIVITY_MULTIPLIER

        # For scroll gesture
        self.scroll_orientation = None
        
        # For Alt+Tab gesture
        self.is_alt_pressed = False  # Whether Alt key is being held
        self.alt_tab_activated = False  # Whether Alt+Tab was activated
        self.y_movement_threshold = DEFAULT_Y_MOVEMENT_THRESHOLD  # Minimum Y movement to trigger Alt+Tab


    def reset(self):
        self.gesture_active = False
        self.confirmed_gesture = False
        self.gesture_confirmation_counter = 0
        self.prev_x = None
        self.prev_y = None
        self.scroll_orientation = None

    def update_cooldown(self):
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1

        if self.gesture_cooldown_counter > 0:
            self.gesture_cooldown_counter -= 1

def is_navigation_gesture(hand_landmarks):
    """
    Check if the hand is making the navigation gesture:
    - Index and middle fingers extended
    - Ring and pinky fingers bent
    """
    # Get fingertips and knuckles
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    index_pip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    middle_pip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    ring_pip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    pinky_pip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP]
    
    # Check if index and middle fingers are extended (tip y < pip y)
    index_extended = index_tip.y < index_pip.y
    middle_extended = middle_tip.y < middle_pip.y
    
    # Check if ring and pinky fingers are bent (tip y > pip y)
    ring_bent = ring_tip.y > ring_pip.y
    pinky_bent = pinky_tip.y > pinky_pip.y
    
    return index_extended and middle_extended and ring_bent and pinky_bent

def is_scroll_gesture(hand_landmarks):
    """
    Check if the hand is making a scroll gesture:
    - Index finger and thumb extended in a V or L shape
    - Other three fingers bent
    
    This is similar to an "L" shape with thumb and index finger
    Slightly relaxed condition for angle acceptance
    """
    # Get fingertips and knuckles
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    thumb_ip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]
    thumb_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    index_pip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP]
    index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    middle_pip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    ring_pip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    pinky_pip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP]
    
    # Calculate distances and positions to check for extended thumb
    thumb_extended = thumb_tip.x < thumb_mcp.x  # For right hand
    
    # Make sure thumb is really extended (distance from wrist)
    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
    dist_thumb_tip_to_wrist = np.sqrt((thumb_tip.x - wrist.x)**2 + (thumb_tip.y - wrist.y)**2)
    dist_thumb_ip_to_wrist = np.sqrt((thumb_ip.x - wrist.x)**2 + (thumb_ip.y - wrist.y)**2)
    thumb_really_extended = dist_thumb_tip_to_wrist > dist_thumb_ip_to_wrist
    
    # Check if index finger is extended (tip further from wrist than pip)
    dist_index_tip_to_wrist = np.sqrt((index_tip.x - wrist.x)**2 + (index_tip.y - wrist.y)**2)
    dist_index_pip_to_wrist = np.sqrt((index_pip.x - wrist.x)**2 + (index_pip.y - wrist.y)**2)
    index_extended = dist_index_tip_to_wrist > dist_index_pip_to_wrist
    
    # Check if other three fingers are bent
    middle_bent = middle_tip.y > middle_pip.y
    ring_bent = ring_tip.y > ring_pip.y
    pinky_bent = pinky_tip.y > pinky_pip.y
    
    # Calculate angle between thumb and index finger to ensure it's a "V" shape
    # First, normalize the vectors from wrist to thumb_tip and wrist to index_tip
    thumb_vec = np.array([thumb_tip.x - wrist.x, thumb_tip.y - wrist.y])
    index_vec = np.array([index_tip.x - wrist.x, index_tip.y - wrist.y])
    
    thumb_vec_norm = thumb_vec / np.linalg.norm(thumb_vec) if np.linalg.norm(thumb_vec) > 0 else thumb_vec
    index_vec_norm = index_vec / np.linalg.norm(index_vec) if np.linalg.norm(index_vec) > 0 else index_vec
    
    # Calculate dot product and angle
    dot_product = np.dot(thumb_vec_norm, index_vec_norm)
    # Clamp the dot product to avoid numerical errors
    dot_product = max(min(dot_product, 1.0), -1.0)
    angle = np.arccos(dot_product) * 180 / np.pi
    
    # V or L shape typically has angle around 45-90+ degrees
    # Only keep the expanded angle range (20-120 degrees instead of 30-110)
    proper_v_shape = angle > 20 and angle < 120
    
    # Ensure there's sufficient separation between thumb and index finger
    thumb_index_distance = np.sqrt(
        (thumb_tip.x - index_tip.x) ** 2 + 
        (thumb_tip.y - index_tip.y) ** 2
    )
    sufficient_separation = thumb_index_distance > 0.05
    
    return (thumb_extended and thumb_really_extended and index_extended and 
            middle_bent and ring_bent and pinky_bent and 
            proper_v_shape and sufficient_separation)

def is_open_hand(hand_landmarks):
    """
    Check if the hand is making an open hand gesture:
    - All five fingers are extended
    """
    # Get fingertips and PIPs (Proximal Interphalangeal Joint)
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    thumb_ip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]
    thumb_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP]
    
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    index_pip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP]
    
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    middle_pip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
    
    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    ring_pip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP]
    
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    pinky_pip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP]
    
    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
    
    # Calculate distances from fingertips to wrist
    dist_thumb_tip_to_wrist = np.sqrt((thumb_tip.x - wrist.x)**2 + (thumb_tip.y - wrist.y)**2)
    dist_thumb_ip_to_wrist = np.sqrt((thumb_ip.x - wrist.x)**2 + (thumb_tip.y - wrist.y)**2)
    
    dist_index_tip_to_wrist = np.sqrt((index_tip.x - wrist.x)**2 + (index_tip.y - wrist.y)**2)
    dist_index_pip_to_wrist = np.sqrt((index_pip.x - wrist.x)**2 + (index_pip.y - wrist.y)**2)
    
    dist_middle_tip_to_wrist = np.sqrt((middle_tip.x - wrist.x)**2 + (middle_tip.y - wrist.y)**2)
    dist_middle_pip_to_wrist = np.sqrt((middle_pip.x - wrist.x)**2 + (middle_pip.y - wrist.y)**2)
    
    dist_ring_tip_to_wrist = np.sqrt((ring_tip.x - wrist.x)**2 + (ring_tip.y - wrist.y)**2)
    dist_ring_pip_to_wrist = np.sqrt((ring_pip.x - wrist.x)**2 + (ring_pip.y - wrist.y)**2)
    
    dist_pinky_tip_to_wrist = np.sqrt((pinky_tip.x - wrist.x)**2 + (pinky_tip.y - wrist.y)**2)
    dist_pinky_pip_to_wrist = np.sqrt((pinky_pip.x - wrist.x)**2 + (pinky_pip.y - wrist.y)**2)
    
    # Check if all fingers are extended
    thumb_extended = dist_thumb_tip_to_wrist > dist_thumb_ip_to_wrist
    index_extended = dist_index_tip_to_wrist > dist_index_pip_to_wrist
    middle_extended = dist_middle_tip_to_wrist > dist_middle_pip_to_wrist
    ring_extended = dist_ring_tip_to_wrist > dist_ring_pip_to_wrist
    pinky_extended = dist_pinky_tip_to_wrist > dist_pinky_pip_to_wrist
    
    # Also check using y-coordinate comparison
    index_extended_y = index_tip.y < index_pip.y
    middle_extended_y = middle_tip.y < middle_pip.y
    ring_extended_y = ring_tip.y < ring_pip.y
    pinky_extended_y = pinky_tip.y < pinky_pip.y
    
    # Combine the checks
    index_is_extended = index_extended and index_extended_y
    middle_is_extended = middle_extended and middle_extended_y
    ring_is_extended = ring_extended and ring_extended_y
    pinky_is_extended = pinky_extended and pinky_extended_y
    
    # All fingers must be extended
    return thumb_extended and index_is_extended and middle_is_extended and ring_is_extended and pinky_is_extended

def is_closed_hand(hand_landmarks):
    """
    This function has been disabled - Alt+F4 gesture feature was removed
    Always returns False
    """
    # Feature disabled
    return False
    middle_bent = middle_tip.y > middle_pip.y
    ring_bent = ring_tip.y > ring_pip.y
    pinky_bent = pinky_tip.y > pinky_pip.y
    
    return index_bent and middle_bent and ring_bent and pinky_bent

def is_index_finger_only(hand_landmarks):
    """
    Check if only the index finger is extended while all other fingers are closed.
    This version has been slightly relaxed to make it easier to perform the gesture.
    """
    # Get landmarks for all fingers and wrist
    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
    
    # Index finger points
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    index_pip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP]
    index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    
    # Middle finger points
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    middle_pip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
    middle_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
    
    # Ring finger points
    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    ring_pip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP]
    ring_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP]
    
    # Pinky points
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    pinky_pip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP]
    pinky_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP]
    
    # Thumb points
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    thumb_ip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]
    thumb_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP]
    
    # Calculate distances from fingertips to wrist
    dist_index_tip_to_wrist = np.sqrt((index_tip.x - wrist.x)**2 + (index_tip.y - wrist.y)**2)
    dist_index_pip_to_wrist = np.sqrt((index_pip.x - wrist.x)**2 + (index_pip.y - wrist.y)**2)
    
    dist_middle_tip_to_wrist = np.sqrt((middle_tip.x - wrist.x)**2 + (middle_tip.y - wrist.y)**2)
    dist_middle_pip_to_wrist = np.sqrt((middle_pip.x - wrist.x)**2 + (middle_pip.y - wrist.y)**2)
    
    dist_ring_tip_to_wrist = np.sqrt((ring_tip.x - wrist.x)**2 + (ring_tip.y - wrist.y)**2)
    dist_ring_pip_to_wrist = np.sqrt((ring_pip.x - wrist.x)**2 + (ring_pip.y - wrist.y)**2)
    
    dist_pinky_tip_to_wrist = np.sqrt((pinky_tip.x - wrist.x)**2 + (pinky_tip.y - wrist.y)**2)
    dist_pinky_pip_to_wrist = np.sqrt((pinky_pip.x - wrist.x)**2 + (pinky_pip.y - wrist.y)**2)
    
    # Check if index finger is extended - slightly relaxed condition
    index_extended = dist_index_tip_to_wrist > dist_index_pip_to_wrist * 0.9
    
    # Check if other fingers are bent
    middle_bent = dist_middle_tip_to_wrist < dist_middle_pip_to_wrist
    ring_bent = dist_ring_tip_to_wrist < dist_ring_pip_to_wrist
    pinky_bent = dist_pinky_tip_to_wrist < dist_pinky_pip_to_wrist
    
    # Calculate distances from thumb tip to other finger tips and bases
    thumb_to_middle_tip = np.sqrt((thumb_tip.x - middle_tip.x)**2 + (thumb_tip.y - middle_tip.y)**2)
    thumb_to_ring_tip = np.sqrt((thumb_tip.x - ring_tip.x)**2 + (thumb_tip.y - ring_tip.y)**2)
    thumb_to_pinky_tip = np.sqrt((thumb_tip.x - pinky_tip.x)**2 + (thumb_tip.y - pinky_tip.y)**2)
    
    # Check if thumb is close to any of the bent fingers
    touch_threshold = 0.05
    thumb_touching_bent_fingers = (
        thumb_to_middle_tip < touch_threshold or
        thumb_to_ring_tip < touch_threshold or
        thumb_to_pinky_tip < touch_threshold or
        # Allow thumb to be positioned near the palm
        (thumb_tip.x > index_mcp.x - 0.05)  # For right-handed users, thumb is inside the palm
    )
    
    # Secondary condition: index fingertip's y coordinate should be higher than others
    index_is_highest = (
        index_tip.y < middle_tip.y - 0.05 and  # Restored to original 0.05
        index_tip.y < ring_tip.y - 0.05 and    # Restored to original 0.05
        index_tip.y < pinky_tip.y - 0.05       # Restored to original 0.05
    )
    
    # Final check: middle is bent
    middle_really_bent = middle_tip.y > middle_pip.y * 0.95
    
    # All conditions must be met
    return (index_extended and 
            middle_bent and ring_bent and pinky_bent and 
            thumb_touching_bent_fingers and 
            middle_really_bent and 
            index_is_highest)

def is_ok_gesture(hand_landmarks):
    """
    Check if the hand is making an "OK" gesture:
    - Index finger and thumb form a circle (tips are close to each other)
    - Other three fingers (middle, ring, pinky) are extended
    """
    # Get finger landmarks
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    index_pip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    middle_pip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    ring_pip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    pinky_pip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP]
    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
    
    # Check if index finger and thumb are forming a circle (tips close to each other)
    # Calculate the distance between index finger tip and thumb tip
    thumb_index_distance = np.sqrt(
        (thumb_tip.x - index_tip.x) ** 2 + 
        (thumb_tip.y - index_tip.y) ** 2 + 
        (thumb_tip.z - index_tip.z) ** 2
    )
    
    # Distance threshold to consider thumb and index are touching
    # This value might need adjustment based on testing
    touching_threshold = 0.05
    
    # Check if middle, ring, and pinky fingers are extended
    # Use a combination of y-coordinate comparison and distance calculation
    middle_extended = middle_tip.y < middle_pip.y
    ring_extended = ring_tip.y < ring_pip.y
    pinky_extended = pinky_tip.y < pinky_pip.y
    
    # Additional checks for finger extension using distance
    dist_middle_tip_to_wrist = np.sqrt((middle_tip.x - wrist.x)**2 + (middle_tip.y - wrist.y)**2)
    dist_middle_pip_to_wrist = np.sqrt((middle_pip.x - wrist.x)**2 + (middle_pip.y - wrist.y)**2)
    dist_ring_tip_to_wrist = np.sqrt((ring_tip.x - wrist.x)**2 + (ring_tip.y - wrist.y)**2)
    dist_ring_pip_to_wrist = np.sqrt((ring_pip.x - wrist.x)**2 + (ring_pip.y - wrist.y)**2)
    dist_pinky_tip_to_wrist = np.sqrt((pinky_tip.x - wrist.x)**2 + (pinky_tip.y - wrist.y)**2)
    dist_pinky_pip_to_wrist = np.sqrt((pinky_pip.x - wrist.x)**2 + (pinky_pip.y - wrist.y)**2)
    
    middle_dist_extended = dist_middle_tip_to_wrist > dist_middle_pip_to_wrist
    ring_dist_extended = dist_ring_tip_to_wrist > dist_ring_pip_to_wrist
    pinky_dist_extended = dist_pinky_tip_to_wrist > dist_pinky_pip_to_wrist
    
    # Final determination combining all checks
    thumb_index_circle = thumb_index_distance < touching_threshold
    other_fingers_extended = (
        (middle_extended and middle_dist_extended) and 
        (ring_extended and ring_dist_extended) and 
        (pinky_extended and pinky_dist_extended)
    )
    
    return thumb_index_circle and other_fingers_extended

def is_alt_tab_ok_gesture(hand_landmarks):
    """
    Check if the hand is making an "OK" gesture for Alt+Tab functionality
    This version is more relaxed than the standard OK gesture:
    - Index finger and thumb form a circle (tips are somewhat close to each other)
    - Only need 2 of 3 other fingers extended
    """
    # Get finger landmarks
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    index_pip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    middle_pip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    ring_pip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    pinky_pip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP]
    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
    
    # Check if index finger and thumb are forming a circle (tips close to each other)
    # Calculate the distance between index finger tip and thumb tip
    thumb_index_distance = np.sqrt(
        (thumb_tip.x - index_tip.x) ** 2 + 
        (thumb_tip.y - index_tip.y) ** 2 + 
        (thumb_tip.z - index_tip.z) ** 2
    )
    
    # More relaxed distance threshold for Alt+Tab OK gesture
    touching_threshold = 0.1  # Increased from 0.05 to 0.1
    
    # Check if middle, ring, and pinky fingers are extended
    # Use a combination of y-coordinate comparison and distance calculation
    middle_extended = middle_tip.y < middle_pip.y * 1.05  # More relaxed condition
    ring_extended = ring_tip.y < ring_pip.y * 1.05  # More relaxed condition
    pinky_extended = pinky_tip.y < pinky_pip.y * 1.05  # More relaxed condition
    
    # Additional checks for finger extension using distance
    dist_middle_tip_to_wrist = np.sqrt((middle_tip.x - wrist.x)**2 + (middle_tip.y - wrist.y)**2)
    dist_middle_pip_to_wrist = np.sqrt((middle_pip.x - wrist.x)**2 + (middle_pip.y - wrist.y)**2)
    dist_ring_tip_to_wrist = np.sqrt((ring_tip.x - wrist.x)**2 + (ring_tip.y - wrist.y)**2)
    dist_ring_pip_to_wrist = np.sqrt((ring_pip.x - wrist.x)**2 + (ring_pip.y - wrist.y)**2)
    dist_pinky_tip_to_wrist = np.sqrt((pinky_tip.x - wrist.x)**2 + (pinky_tip.y - wrist.y)**2)
    dist_pinky_pip_to_wrist = np.sqrt((pinky_pip.x - wrist.x)**2 + (pinky_pip.y - wrist.y)**2)
    
    middle_dist_extended = dist_middle_tip_to_wrist > dist_middle_pip_to_wrist * 0.9  # More relaxed
    ring_dist_extended = dist_ring_tip_to_wrist > dist_ring_pip_to_wrist * 0.9  # More relaxed
    pinky_dist_extended = dist_pinky_tip_to_wrist > dist_pinky_pip_to_wrist * 0.9  # More relaxed
    
    # Final determination with more relaxed conditions
    thumb_index_circle = thumb_index_distance < touching_threshold
    
    # Only 2 out of 3 fingers need to be extended (more relaxed)
    middle_extended_final = middle_extended or middle_dist_extended
    ring_extended_final = ring_extended or ring_dist_extended
    pinky_extended_final = pinky_extended or pinky_dist_extended
    
    # Count extended fingers
    extended_count = sum([middle_extended_final, ring_extended_final, pinky_extended_final])
    other_fingers_extended = extended_count >= 2  # Only need 2 out of 3 fingers extended
    
    return thumb_index_circle and other_fingers_extended

def update_all_cooldowns(gesture_states, fps):
    """Update all gesture state cooldowns"""
    for state in gesture_states.values():
        state.update_cooldown()

def reset_all_gesture_states(gesture_states):
    """Reset all gesture states when no hand is detected"""
    for state in gesture_states.values():
        state.reset()