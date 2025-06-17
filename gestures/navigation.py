#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Navigation gesture module for controlling left/right arrow keys
"""

import cv2
import time
from pynput.keyboard import Key
from config import keyboard
from gestures.base import GestureState, is_navigation_gesture

# Create state for navigation gesture
navigation_state = GestureState()

def process_navigation_gesture(image, hand_landmarks, fps, image_width, image_height):
    """Process navigation gesture (index and middle finger extended) to control left/right keys"""
    global navigation_state
    state = navigation_state
    
    # Check if we're in cooldown period after a successful navigation gesture
    if state.is_cooldown_active():
        current_time = time.time()
        remaining_cooldown = max(0, state.gesture_cooldown_time - (current_time - state.gesture_cooldown_start_time))
        
        # Display cooldown message
        cv2.putText(
            image, 
            f"Navigation cooldown: {remaining_cooldown:.1f}s", 
            (10, 70), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, 
            (255, 165, 0), 
            2
        )
        
        # Update state and return early
        navigation_state = state
        return image
    
    # Check if gesture is still valid
    is_valid_gesture = is_navigation_gesture(hand_landmarks)
    
    # If gesture was confirmed but is no longer valid, cancel it
    if state.confirmed_gesture and not is_valid_gesture:
        state.confirmed_gesture = False
        state.gesture_confirmation_start_time = 0
        state.prev_x = None
        
        # Display cancellation message
        cv2.putText(
            image, 
            "Navigation gesture cancelled: Invalid gesture", 
            (10, 70), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, 
            (0, 0, 255), 
            2
        )
        
        # Update state and return
        navigation_state = state
        return image
    
    # If gesture was confirmed, track movement
    if state.confirmed_gesture:
        index_tip = hand_landmarks.landmark[8]  # INDEX_FINGER_TIP
        middle_tip = hand_landmarks.landmark[12]  # MIDDLE_FINGER_TIP
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
          # Only track movement if we have a previous position and not in brief cooldown
        current_time = time.time()
        if state.prev_x is not None and not (current_time - state.cooldown_start_time < state.cooldown_duration):
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
                    
                    # Start cooldown periods after action
                    state.start_cooldown(0.2)  # Brief cooldown after key press                    state.start_gesture_cooldown()  # Main cooldown for gesture 
                    
                    # Reset the gesture detection process
                    state.confirmed_gesture = False
                    state.gesture_confirmation_start_time = 0
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
                    
                    # Start cooldown periods after action
                    state.start_cooldown(0.2)  # Brief cooldown after key press
                    state.start_gesture_cooldown()  # Main cooldown for gesture
                    
                    # Reset the gesture detection process
                    state.confirmed_gesture = False
                    state.gesture_confirmation_start_time = 0
                    state.prev_x = None
        
        # Update previous position
        state.prev_x = current_x
        state.gesture_active = True
          # If gesture is not yet confirmed but is valid, start or continue confirmation timer
    elif is_valid_gesture:
        # Start confirmation timer if not started
        if state.gesture_confirmation_start_time == 0:
            state.start_gesture_confirmation()
        
        # Calculate confirmation progress
        current_time = time.time()
        elapsed_time = current_time - state.gesture_confirmation_start_time
        confirmation_percent = min(100, int((elapsed_time / state.gesture_confirmation_time) * 100))
        
        # Display confirmation progress
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
        if state.is_gesture_confirmed():
            state.confirmed_gesture = True
            # Get initial position once gesture is confirmed
            index_tip = hand_landmarks.landmark[8]  # INDEX_FINGER_TIP
            middle_tip = hand_landmarks.landmark[12]  # MIDDLE_FINGER_TIP
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
    else:
        # Reset confirmation timer if gesture is not valid
        state.gesture_confirmation_start_time = 0
    
    # Update state
    navigation_state = state
    return image