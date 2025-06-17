#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Mouse control gesture module for controlling cursor using index finger
"""

import cv2
import time
from config import mouse
from gestures.base import GestureState

# Create state for mouse control gesture
mouse_state = GestureState()

def process_mouse_control_gesture(image, hand_landmarks, fps, image_width, image_height):
    """Process index finger only gesture to control the mouse cursor"""
    global mouse_state
    state = mouse_state
    
    # Check if gesture is still valid (is_index_finger_only needs to be imported)
    from gestures.base import is_index_finger_only
    is_valid_gesture = is_index_finger_only(hand_landmarks)
    
    # If gesture was confirmed but is no longer valid, cancel it
    if state.confirmed_gesture and not is_valid_gesture:
        state.confirmed_gesture = False
        state.gesture_confirmation_start_time = 0
        
        cv2.putText(
            image, 
            "Mouse control canceled", 
            (10, 70), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, 
            (0, 0, 255), 
            2
        )
        
        # Update state and return
        mouse_state = state
        return image

    # Process confirmed gesture
    if state.confirmed_gesture and is_valid_gesture:
        # Get index finger tip position
        index_tip = hand_landmarks.landmark[8]  # INDEX_FINGER_TIP
        
        # Convert to screen coordinates
        current_x = index_tip.x * image_width
        current_y = index_tip.y * image_height
        
        # Draw cursor indicator
        cv2.circle(image, (int(current_x), int(current_y)), 10, (0, 255, 0), -1)
        cv2.putText(
            image, 
            "CURSOR", 
            (int(current_x) - 20, int(current_y) - 15), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.5, 
            (0, 255, 0), 
            2
        )
        
        cv2.putText(
            image, 
            "Mouse Control Active", 
            (10, 70), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, 
            (0, 255, 0), 
            2
        )
        
        # Convert gesture coordinates to screen coordinates
        # Use direct mapping for natural control
        screen_x = index_tip.x * state.screen_width
        screen_y = index_tip.y * state.screen_height
        
        # Apply smoothing for more stable cursor movement
        if hasattr(state, 'smooth_x') and hasattr(state, 'smooth_y'):
            state.smooth_x = state.smooth_x * state.smoothing_factor + screen_x * (1 - state.smoothing_factor)
            state.smooth_y = state.smooth_y * state.smoothing_factor + screen_y * (1 - state.smoothing_factor)
        else:
            state.smooth_x = screen_x
            state.smooth_y = screen_y
        
        # Apply sensitivity multiplier
        final_x = state.smooth_x * state.sensitivity_multiplier
        final_y = state.smooth_y * state.sensitivity_multiplier
        
        # Ensure coordinates are within screen bounds
        final_x = max(0, min(state.screen_width - 1, final_x))
        final_y = max(0, min(state.screen_height - 1, final_y))
        
        # Move mouse cursor
        mouse.position = (final_x, final_y)
        
        # Display coordinates
        cv2.putText(
            image, 
            f"Screen: ({final_x:.0f}, {final_y:.0f})", 
            (10, 110), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, 
            (255, 255, 255), 
            2
        )
        
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
        
        # Get index finger tip position for preview
        index_tip = hand_landmarks.landmark[8]
        preview_x = index_tip.x * image_width
        preview_y = index_tip.y * image_height
        
        # Draw preview cursor (orange color)
        cv2.circle(image, (int(preview_x), int(preview_y)), 8, (0, 165, 255), -1)
        cv2.putText(
            image, 
            "PREVIEW", 
            (int(preview_x) - 25, int(preview_y) - 15), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.4, 
            (0, 165, 255), 
            1
        )
        
        # Display confirmation progress
        cv2.putText(
            image, 
            f"Confirming mouse control: {confirmation_percent}%", 
            (10, 70), 
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
                "Mouse Control Confirmed!", 
                (10, 110), 
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
                "Mouse control ended", 
                (10, 70), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                (0, 255, 255), 
                2
            )
            
            # Reset mouse control state
            state.reset()
    
    # Update state
    mouse_state = state
    return image
