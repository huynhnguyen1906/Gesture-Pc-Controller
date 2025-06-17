#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Mouse clicking gesture module for clicking the mouse while controlling the cursor
"""

import cv2
import time
from config import mouse
from pynput.mouse import Button
from gestures.base import GestureState

# Create state for mouse click gesture
mouse_click_state = GestureState()

def process_mouse_click_gesture(image, hand_landmarks, fps, image_width, image_height):
    """Process OK gesture to trigger mouse clicks"""
    global mouse_click_state
    state = mouse_click_state
    
    # Check if gesture is still valid (is_ok_gesture needs to be imported)
    from gestures.base import is_ok_gesture
    is_valid_gesture = is_ok_gesture(hand_landmarks)
    
    # If gesture was confirmed but is no longer valid, cancel it
    if state.confirmed_gesture and not is_valid_gesture:
        state.confirmed_gesture = False
        state.gesture_confirmation_start_time = 0
        
        cv2.putText(
            image, 
            "Mouse click canceled", 
            (10, 150), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, 
            (0, 0, 255), 
            2
        )
        
        # Update state and return
        mouse_click_state = state
        return image

    # Process confirmed gesture
    if state.confirmed_gesture and is_valid_gesture:
        # Check if we're in cooldown after a recent click
        current_time = time.time()
        if current_time - state.cooldown_start_time < state.cooldown_duration:
            remaining_cooldown = state.cooldown_duration - (current_time - state.cooldown_start_time)
            
            cv2.putText(
                image, 
                f"Click cooldown: {remaining_cooldown:.1f}s", 
                (10, 150), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                (255, 165, 0), 
                2
            )
        else:
            # Perform click
            mouse.click(Button.left)
            
            cv2.putText(
                image, 
                "LEFT CLICK!", 
                (10, 150), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.9, 
                (0, 0, 255), 
                2
            )
            
            # Start cooldown period
            state.start_cooldown(0.5)  # 0.5 second cooldown
            
            # Reset gesture after click
            state.confirmed_gesture = False
            state.gesture_confirmation_start_time = 0
        
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
        
        # Get thumb and index finger tips for visualization
        thumb_tip = hand_landmarks.landmark[4]  # THUMB_TIP
        index_tip = hand_landmarks.landmark[8]  # INDEX_FINGER_TIP
        
        thumb_x = int(thumb_tip.x * image_width)
        thumb_y = int(thumb_tip.y * image_height)
        index_x = int(index_tip.x * image_width)
        index_y = int(index_tip.y * image_height)
        
        # Draw OK gesture indicator
        cv2.circle(image, (thumb_x, thumb_y), 8, (255, 0, 255), -1)
        cv2.circle(image, (index_x, index_y), 8, (255, 0, 255), -1)
        cv2.line(image, (thumb_x, thumb_y), (index_x, index_y), (255, 0, 255), 3)
        
        cv2.putText(
            image, 
            "OK", 
            ((thumb_x + index_x) // 2 - 10, (thumb_y + index_y) // 2 - 10), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.6, 
            (255, 0, 255), 
            2
        )
        
        # Display confirmation progress
        cv2.putText(
            image, 
            f"Confirming mouse click: {confirmation_percent}%", 
            (10, 150), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, 
            (255, 255, 0), 
            2
        )
        
        # Check if gesture has been held long enough to confirm
        if state.is_gesture_confirmed():
            state.confirmed_gesture = True
            
            cv2.putText(
                image, 
                "Mouse Click Confirmed!", 
                (10, 190), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                (0, 255, 0), 
                2
            )
    else:
        # Reset confirmation timer if gesture is not valid
        state.gesture_confirmation_start_time = 0
        
        # If we were in the middle of an active gesture, reset it
        if state.confirmed_gesture:
            cv2.putText(
                image, 
                "Mouse click ended", 
                (10, 150), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                (0, 255, 255), 
                2
            )
            
            # Reset click state
            state.reset()
    
    # Update state
    mouse_click_state = state
    return image
