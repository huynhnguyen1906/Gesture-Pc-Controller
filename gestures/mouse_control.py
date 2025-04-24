#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Mouse control gesture module for controlling cursor using index finger
"""

import cv2
from config import mouse
from gestures.base import GestureState

# Create state for mouse control gesture
mouse_state = GestureState()

def process_mouse_control_gesture(image, hand_landmarks, fps, image_width, image_height):
    """Process index finger only gesture to control the mouse cursor"""
    global mouse_state
    state = mouse_state
    
    # Calculate frames needed for gesture confirmation and cooldown
    confirmation_frames = int(state.gesture_confirmation_time * fps)
    
    # Check if gesture is still valid (is_index_finger_only needs to be imported)
    from gestures.base import is_index_finger_only
    is_valid_gesture = is_index_finger_only(hand_landmarks)
    
    # If gesture was confirmed but is no longer valid, cancel it
    if state.confirmed_gesture and not is_valid_gesture:
        state.confirmed_gesture = False
        state.gesture_confirmation_counter = 0
        state.prev_x = None
        state.prev_y = None
        
        # Display cancellation message
        cv2.putText(
            image, 
            "Mouse control cancelled: Invalid gesture", 
            (10, 390), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, 
            (0, 0, 255), 
            2
        )
        
        # Update state and return early
        mouse_state = state
        return image
    
    # If gesture was confirmed, track movement
    if state.confirmed_gesture:
        # Get index finger tip position
        index_tip = hand_landmarks.landmark[8]  # INDEX_FINGER_TIP
        
        # Convert from normalized coordinates to pixel coordinates
        index_tip_x = index_tip.x * image_width
        index_tip_y = index_tip.y * image_height
        
        # Draw cursor control indicator
        cv2.circle(image, (int(index_tip_x), int(index_tip_y)), 15, (0, 0, 255), -1)
        cv2.circle(image, (int(index_tip_x), int(index_tip_y)), 15, (255, 255, 255), 2)
        
        # Display active gesture status
        cv2.putText(
            image, 
            "Mouse Control Active", 
            (10, 390), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, 
            (0, 255, 0), 
            2
        )
        
        # Only update mouse position if we have previous values
        if state.prev_x is not None and state.prev_y is not None:
            # Calculate mouse position with smoothing
            # Convert finger position from camera coordinates to screen coordinates
            # Use a mapping where:
            # - Camera's 20%-80% space maps to the full screen to avoid edges
            # - Smaller movements in the camera translate to larger movements on screen
            
            # Get current mouse position
            current_mouse_x, current_mouse_y = mouse.position
            
            # Map finger position from camera to screen
            camera_x_min = image_width * 0.2  # 20% from left edge
            camera_x_max = image_width * 0.8  # 20% from right edge
            camera_y_min = image_height * 0.2  # 20% from top edge
            camera_y_max = image_height * 0.8  # 20% from bottom edge
            
            # Constrain finger position within camera mapping area
            constrained_x = max(camera_x_min, min(camera_x_max, index_tip_x))
            constrained_y = max(camera_y_min, min(camera_y_max, index_tip_y))
            
            # Calculate normalized position (0-1 range)
            normalized_x = (constrained_x - camera_x_min) / (camera_x_max - camera_x_min)
            normalized_y = (constrained_y - camera_y_min) / (camera_y_max - camera_y_min)
            
            # Map to screen coordinates with sensitivity multiplier
            target_mouse_x = int(normalized_x * state.screen_width * state.sensitivity_multiplier)
            target_mouse_y = int(normalized_y * state.screen_height * state.sensitivity_multiplier)
            
            # Apply smoothing
            smoothed_x = int(current_mouse_x + (target_mouse_x - current_mouse_x) * (1 - state.smoothing_factor))
            smoothed_y = int(current_mouse_y + (target_mouse_y - current_mouse_y) * (1 - state.smoothing_factor))
            
            # Move mouse to new position
            mouse.position = (smoothed_x, smoothed_y)
            
            # Display mouse coordinates
            cv2.putText(
                image, 
                f"Mouse: ({smoothed_x}, {smoothed_y})", 
                (10, 430), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                (255, 0, 0), 
                2
            )
        
        # Update previous finger position
        state.prev_x = index_tip_x
        state.prev_y = index_tip_y
        state.gesture_active = True
        
    # If gesture is not yet confirmed but is valid, increment confirmation counter
    elif is_valid_gesture:
        # Increment confirmation counter
        state.gesture_confirmation_counter += 1
        
        # Display confirmation progress
        confirmation_percent = min(100, int((state.gesture_confirmation_counter / confirmation_frames) * 100))
        cv2.putText(
            image, 
            f"Confirming mouse control gesture: {confirmation_percent}%", 
            (10, 390), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, 
            (255, 255, 0), 
            2
        )
        
        # Check if gesture has been held long enough to confirm
        if state.gesture_confirmation_counter >= confirmation_frames:
            state.confirmed_gesture = True
            # Get initial position once gesture is confirmed
            index_tip = hand_landmarks.landmark[8]  # INDEX_FINGER_TIP
            state.prev_x = index_tip.x * image_width
            state.prev_y = index_tip.y * image_height
            
            # Get screen resolution (could be done once at startup instead)
            import screeninfo
            try:
                screen = screeninfo.get_monitors()[0]
                state.screen_width = screen.width
                state.screen_height = screen.height
            except:
                # Fallback to default values if screeninfo is not available
                state.screen_width = 1920
                state.screen_height = 1080
            
            cv2.putText(
                image, 
                "Mouse Control Activated!", 
                (10, 430), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                (0, 255, 0), 
                2
            )
    else:
        # Reset confirmation counter if gesture is not valid
        state.gesture_confirmation_counter = 0
    
    # Update state
    mouse_state = state
    return image