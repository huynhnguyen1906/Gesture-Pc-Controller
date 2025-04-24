#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Mouse clicking gesture module for clicking the mouse while controlling the cursor
"""

import cv2
from config import mouse
from pynput.mouse import Button  # Thêm import Button từ pynput.mouse
from gestures.base import GestureState

# Create state for mouse click gesture
mouse_click_state = GestureState()

def process_mouse_click_gesture(image, hand_landmarks, fps, image_width, image_height):
    """Process OK gesture to trigger mouse clicks"""
    global mouse_click_state
    state = mouse_click_state
    
    # Calculate frames needed for gesture confirmation and cooldown
    confirmation_frames = int(state.gesture_confirmation_time * fps)
    
    # Check if gesture is still valid (is_ok_gesture needs to be imported)
    from gestures.base import is_ok_gesture
    is_valid_gesture = is_ok_gesture(hand_landmarks)
    
    # If gesture was confirmed but is no longer valid, cancel it
    if state.confirmed_gesture and not is_valid_gesture:
        state.confirmed_gesture = False
        state.gesture_confirmation_counter = 0
        
        # Display cancellation message
        cv2.putText(
            image, 
            "Mouse click cancelled: Invalid gesture", 
            (10, 470), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, 
            (0, 0, 255), 
            2
        )
        
        # Update state and return early
        mouse_click_state = state
        return image
    
    # If gesture was confirmed, perform the mouse click
    if state.confirmed_gesture:
        if state.cooldown_counter == 0:
            # Perform left mouse click - sử dụng Button.left thay vì string 'left'
            mouse.click(Button.left)
            
            # Display click status
            cv2.putText(
                image, 
                "LEFT CLICK!", 
                (10, 510), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.9, 
                (0, 0, 255), 
                2
            )
            
            # Set cooldown to avoid multiple clicks
            state.cooldown_counter = int(0.5 * fps)  # 0.5 second cooldown
        else:
            # Display cooldown status
            cv2.putText(
                image, 
                f"Click cooldown: {state.cooldown_counter / fps:.1f}s", 
                (10, 510), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                (255, 165, 0), 
                2
            )
        
        # Display active gesture status
        cv2.putText(
            image, 
            "Mouse Click Gesture Active", 
            (10, 470), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, 
            (0, 255, 0), 
            2
        )
        
        state.gesture_active = True
        
    # If gesture is not yet confirmed but is valid, increment confirmation counter
    elif is_valid_gesture:
        # Increment confirmation counter
        state.gesture_confirmation_counter += 1
        
        # Display confirmation progress
        confirmation_percent = min(100, int((state.gesture_confirmation_counter / confirmation_frames) * 100))
        cv2.putText(
            image, 
            f"Confirming click gesture: {confirmation_percent}%", 
            (10, 470), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, 
            (255, 255, 0), 
            2
        )
        
        # Check if gesture has been held long enough to confirm
        if state.gesture_confirmation_counter >= confirmation_frames:
            state.confirmed_gesture = True
            
            cv2.putText(
                image, 
                "Click Gesture Activated!", 
                (10, 510), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                (0, 255, 0), 
                2
            )
    else:
        # Reset confirmation counter if gesture is not valid
        state.gesture_confirmation_counter = 0
    
    # Update state
    mouse_click_state = state
    return image