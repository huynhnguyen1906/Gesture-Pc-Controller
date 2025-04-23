#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Gesture detection module for PC Controller
Contains various hand gesture recognizers and their actions
"""

import time
import cv2
import mediapipe as mp
from pynput.keyboard import Key, Controller, KeyCode

# Initialize MediaPipe hands module
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Initialize keyboard controller
keyboard = Controller()

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

# Gesture state tracking variables
class GestureState:
    def __init__(self):
        self.prev_x = None
        self.gesture_active = False
        self.movement_threshold = 30  # Minimum X movement to trigger a keystroke
        self.cooldown_counter = 0
        self.cooldown_frames = 10  # Wait this many frames after a key press
        
        self.gesture_confirmation_time = 0.1  # Seconds to confirm gesture before tracking
        self.gesture_confirmation_counter = 0
        self.gesture_cooldown_time = 0.15  # Seconds between gesture detections
        self.gesture_cooldown_counter = 0
        self.confirmed_gesture = False
        
        # For open-close hand gesture
        self.hand_open = False
        self.hand_closed = False
        self.open_hand_confirmed = False
        self.last_hand_state_change_time = 0

# Initialize gesture states
navigation_gesture_state = GestureState()
alt_f4_gesture_state = GestureState()

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

def is_open_hand(hand_landmarks):
    """
    Check if all five fingers are extended (open hand)
    """
    # Get fingertips and knuckles
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    thumb_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    index_pip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    middle_pip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    ring_pip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    pinky_pip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP]
    
    # Check if all fingers are extended
    thumb_extended = thumb_tip.x < thumb_mcp.x  # For right hand
    index_extended = index_tip.y < index_pip.y
    middle_extended = middle_tip.y < middle_pip.y
    ring_extended = ring_tip.y < ring_pip.y
    pinky_extended = pinky_tip.y < pinky_pip.y
    
    return thumb_extended and index_extended and middle_extended and ring_extended and pinky_extended

def is_closed_hand(hand_landmarks):
    """
    Check if hand is closed (fist)
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
    
    # Check if all fingers are bent (tip y > pip y)
    index_bent = index_tip.y > index_pip.y
    middle_bent = middle_tip.y > middle_pip.y
    ring_bent = ring_tip.y > ring_pip.y
    pinky_bent = pinky_tip.y > pinky_pip.y
    
    return index_bent and middle_bent and ring_bent and pinky_bent

def process_navigation_gesture(image, hand_landmarks, fps, image_width, image_height):
    """Process navigation gesture (index and middle finger extended) to control left/right keys"""
    global navigation_gesture_state
    state = navigation_gesture_state
    
    # Calculate frames needed for gesture confirmation and cooldown
    confirmation_frames = int(state.gesture_confirmation_time * fps)
    cooldown_frames = int(state.gesture_cooldown_time * fps)
    
    # If gesture was confirmed, track movement
    if state.confirmed_gesture:
        index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        current_x = (index_tip.x + middle_tip.x) / 2 * image_width
        
        # Display active gesture status
        cv2.putText(
            image, 
            "Navigation Gesture Active", 
            (10, 70), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, 
            (0, 255, 0), 
            2
        )
        
        # Only track movement if we have a previous position
        if state.prev_x is not None and state.cooldown_counter == 0:
            x_movement = state.prev_x - current_x
            
            # Display movement direction and magnitude
            cv2.putText(
                image, 
                f"Move: {x_movement:.1f}", 
                (10, 110), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                (255, 0, 0), 
                2
            )
            
            # Determine gesture direction for significant movements
            if abs(x_movement) > state.movement_threshold:
                if x_movement > 0:  # Right to left movement
                    cv2.putText(
                        image, 
                        "RIGHT key pressed", 
                        (10, 150), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.9, 
                        (0, 0, 255), 
                        2
                    )
                    keyboard.press(Key.right)
                    keyboard.release(Key.right)
                    state.cooldown_counter = int(0.2 * fps)  # Brief cooldown after key press
                    
                    # Reset the gesture detection process and start cooldown
                    state.confirmed_gesture = False
                    state.gesture_confirmation_counter = 0
                    state.gesture_cooldown_counter = cooldown_frames
                    state.prev_x = None
                elif x_movement < 0:  # Left to right movement
                    cv2.putText(
                        image, 
                        "LEFT key pressed", 
                        (10, 150), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.9, 
                        (0, 0, 255), 
                        2
                    )
                    keyboard.press(Key.left)
                    keyboard.release(Key.left)
                    state.cooldown_counter = int(0.2 * fps)  # Brief cooldown after key press
                    
                    # Reset the gesture detection process and start cooldown
                    state.confirmed_gesture = False
                    state.gesture_confirmation_counter = 0
                    state.gesture_cooldown_counter = cooldown_frames
                    state.prev_x = None
        
        # Update previous position
        state.prev_x = current_x
        state.gesture_active = True
        
    # If gesture is not yet confirmed, increment confirmation counter
    else:
        # Increment confirmation counter
        state.gesture_confirmation_counter += 1
        
        # Display confirmation progress
        confirmation_percent = min(100, int((state.gesture_confirmation_counter / confirmation_frames) * 100))
        cv2.putText(
            image, 
            f"Confirming navigation gesture: {confirmation_percent}%", 
            (10, 70), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, 
            (255, 255, 0), 
            2
        )
        
        # Check if gesture has been held long enough to confirm
        if state.gesture_confirmation_counter >= confirmation_frames:
            state.confirmed_gesture = True
            # Get initial position once gesture is confirmed
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            state.prev_x = (index_tip.x + middle_tip.x) / 2 * image_width
            
            cv2.putText(
                image, 
                "Navigation Gesture Confirmed!", 
                (10, 110), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                (0, 255, 0), 
                2
            )
    
    # Update state
    navigation_gesture_state = state
    return image

def process_alt_f4_gesture(image, hand_landmarks, fps, image_width, image_height):
    """Process open hand to closed hand gesture to trigger Alt+F4"""
    global alt_f4_gesture_state
    state = alt_f4_gesture_state
    
    current_time = time.time()
    is_open = is_open_hand(hand_landmarks)
    is_closed = is_closed_hand(hand_landmarks)
    
    # Calculate frames needed for gesture confirmation
    confirmation_frames = int(state.gesture_confirmation_time * fps)
    
    if is_open and not state.open_hand_confirmed:
        # Increment confirmation counter for open hand
        state.gesture_confirmation_counter += 1
        
        # Display confirmation progress for open hand
        confirmation_percent = min(100, int((state.gesture_confirmation_counter / confirmation_frames) * 100))
        cv2.putText(
            image, 
            f"Confirming open hand: {confirmation_percent}%", 
            (10, 230), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, 
            (255, 255, 0), 
            2
        )
        
        # Check if open hand has been held long enough to confirm
        if state.gesture_confirmation_counter >= confirmation_frames:
            state.open_hand_confirmed = True
            state.last_hand_state_change_time = current_time
            state.gesture_confirmation_counter = 0  # Reset for closed hand detection
            
            cv2.putText(
                image, 
                "Open hand confirmed! Now close hand for Alt+F4", 
                (10, 270), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                (0, 255, 0), 
                2
            )
    
    # If open hand was confirmed, now wait for closed hand
    elif state.open_hand_confirmed and is_closed:
        # Increment confirmation counter for closed hand
        state.gesture_confirmation_counter += 1
        
        # Display confirmation progress for closed hand
        confirmation_percent = min(100, int((state.gesture_confirmation_counter / confirmation_frames) * 100))
        cv2.putText(
            image, 
            f"Confirming closed hand: {confirmation_percent}%", 
            (10, 270), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, 
            (255, 255, 0), 
            2
        )
        
        # Check if closed hand has been held long enough to confirm
        if state.gesture_confirmation_counter >= confirmation_frames:
            # Sequence completed - trigger Alt+F4
            cv2.putText(
                image, 
                "Alt+F4 triggered!", 
                (10, 310), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.9, 
                (0, 0, 255), 
                2
            )
            
            # Execute Alt+F4
            keyboard.press(Key.alt)
            keyboard.press(KeyCode.from_vk(0x73))  # F4 key
            keyboard.release(KeyCode.from_vk(0x73))
            keyboard.release(Key.alt)
            
            # Reset state
            state.open_hand_confirmed = False
            state.gesture_confirmation_counter = 0
            state.gesture_cooldown_counter = int(state.gesture_cooldown_time * fps)
    
    # If timeout or gesture changed, reset the state
    elif state.open_hand_confirmed:
        # Check for timeout (3 seconds to complete the gesture)
        if current_time - state.last_hand_state_change_time > 3.0:
            state.open_hand_confirmed = False
            state.gesture_confirmation_counter = 0
            
            cv2.putText(
                image, 
                "Open-close gesture timed out", 
                (10, 310), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                (0, 0, 255), 
                2
            )
        else:
            # Still waiting for closed hand
            cv2.putText(
                image, 
                "Waiting for closed hand...", 
                (10, 270), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                (255, 165, 0), 
                2
            )
            # Reset confirmation counter if hand is not closed
            state.gesture_confirmation_counter = 0
    
    # If not in open hand confirmation and not in open-closed sequence
    elif not is_open:
        # Reset open hand confirmation counter
        state.gesture_confirmation_counter = 0
    
    # Update cooldown counter
    if state.gesture_cooldown_counter > 0:
        state.gesture_cooldown_counter -= 1
        # Display cooldown message
        cv2.putText(
            image, 
            f"Alt+F4 gesture cooldown: {state.gesture_cooldown_counter / fps:.1f}s", 
            (10, 350), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, 
            (255, 165, 0), 
            2
        )
    
    # Update state
    alt_f4_gesture_state = state
    return image

def reset_gesture_states():
    """Reset all gesture states - call when no hand is detected"""
    global navigation_gesture_state, alt_f4_gesture_state
    
    # Reset navigation gesture
    if navigation_gesture_state.confirmed_gesture:
        navigation_gesture_state.confirmed_gesture = False
        navigation_gesture_state.prev_x = None
    
    navigation_gesture_state.gesture_confirmation_counter = 0
    navigation_gesture_state.gesture_active = False
    
    # Reset alt+f4 gesture
    if alt_f4_gesture_state.open_hand_confirmed:
        alt_f4_gesture_state.open_hand_confirmed = False
    
    alt_f4_gesture_state.gesture_confirmation_counter = 0

def update_cooldowns(fps):
    """Update cooldown counters"""
    global navigation_gesture_state, alt_f4_gesture_state
    
    # Update navigation gesture cooldowns
    if navigation_gesture_state.cooldown_counter > 0:
        navigation_gesture_state.cooldown_counter -= 1
    
    if navigation_gesture_state.gesture_cooldown_counter > 0:
        navigation_gesture_state.gesture_cooldown_counter -= 1
    
    # Update Alt+F4 gesture cooldowns  
    if alt_f4_gesture_state.cooldown_counter > 0:
        alt_f4_gesture_state.cooldown_counter -= 1
    
    if alt_f4_gesture_state.gesture_cooldown_counter > 0:
        alt_f4_gesture_state.gesture_cooldown_counter -= 1