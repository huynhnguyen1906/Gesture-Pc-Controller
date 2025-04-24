#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Scroll gesture module for controlling scrolling with thumb and index finger in L/V shape
Only allows vertical scrolling with reference point tracking
"""

import cv2
import numpy as np
from pynput.mouse import Button, Controller
from config import mouse
from gestures.base import GestureState, is_scroll_gesture

# Create state for scroll gesture
scroll_state = GestureState()

def process_scroll_gesture(image, hand_landmarks, fps, image_width, image_height):
    """
    Process scroll gesture (thumb and index finger extended in L/V shape, other fingers bent)
    to control vertical scrolling only, with reference point visualization
    """
    global scroll_state
    state = scroll_state
    
    # Calculate frames needed for gesture confirmation and cooldown
    confirmation_frames = int(state.gesture_confirmation_time * fps)
    cooldown_frames = int(state.gesture_cooldown_time * fps)
    
    # Check if we're in cooldown period after a successful scroll gesture
    if state.gesture_cooldown_counter > 0:
        # Display cooldown message
        cv2.putText(
            image, 
            f"Scroll cooldown: {state.gesture_cooldown_counter / fps:.1f}s", 
            (10, 310), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, 
            (255, 165, 0), 
            2
        )
        
        # Update state and return early
        scroll_state = state
        return image
    
    # Check if gesture is still valid
    is_valid_gesture = is_scroll_gesture(hand_landmarks)
    
    # If gesture was confirmed but is no longer valid, cancel it and reset reference point
    # Đây là trường hợp khi tay vẫn trong camera nhưng tư thế không còn hợp lệ cho scroll nữa
    if state.confirmed_gesture and not is_valid_gesture:
        state.confirmed_gesture = False
        state.gesture_confirmation_counter = 0
        state.prev_x = None  # Reset reference point
        state.prev_y = None  # Reset reference point
        
        # Display cancellation message
        cv2.putText(
            image, 
            "Scroll gesture cancelled: Invalid gesture", 
            (10, 310), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, 
            (0, 0, 255), 
            2
        )
        
        # Update state and return
        scroll_state = state
        return image
    
    # If gesture was confirmed, track movement
    if state.confirmed_gesture:
        # Get index finger tip and thumb tip positions
        index_tip = hand_landmarks.landmark[8]  # INDEX_FINGER_TIP
        thumb_tip = hand_landmarks.landmark[4]  # THUMB_TIP
        
        # Calculate center point between index and thumb tips (as a reference point)
        current_x = (index_tip.x + thumb_tip.x) / 2 * image_width
        current_y = (index_tip.y + thumb_tip.y) / 2 * image_height
        
        # Draw the current position
        cv2.circle(
            image,
            (int(current_x), int(current_y)),
            5,
            (0, 0, 255),  # Red color for current position
            -1  # Filled circle
        )
        
        # Draw the reference/starting position
        if state.prev_x is not None and state.prev_y is not None:
            # Draw reference point (starting position)
            cv2.circle(
                image,
                (int(state.prev_x), int(state.prev_y)),
                8,
                (0, 255, 0),  # Green color for reference position
                2  # Circle outline
            )
            
            # Draw a line connecting reference point and current point
            cv2.line(
                image,
                (int(state.prev_x), int(state.prev_y)),
                (int(current_x), int(current_y)),
                (255, 255, 0),  # Yellow line
                2
            )
            
            # Calculate vertical distance from reference point
            y_distance = current_y - state.prev_y
            
            # Display distance
            cv2.putText(
                image,
                f"Distance: {int(y_distance)}px",
                (10, 350),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2
            )
            
            # Apply scrolling based on vertical movement
            if state.cooldown_counter == 0:
                # Calculate scroll amount, more movement = faster scrolling
                scroll_amount = int(y_distance / 20)
                
                if abs(scroll_amount) > 0:
                    # Perform scroll action
                    mouse.scroll(0, -scroll_amount)
                    
                    # Display scrolling information
                    direction = "DOWN" if scroll_amount > 0 else "UP"
                    cv2.putText(
                        image,
                        f"Scrolling {direction}: {abs(scroll_amount)}",
                        (10, 390),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2
                    )
                    
                    # Set a small cooldown to prevent too many scroll events
                    state.cooldown_counter = max(1, int(0.02 * fps))
        
        # Display active gesture status
        cv2.putText(
            image, 
            "Scroll Gesture Active", 
            (10, 310), 
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
            f"Confirming scroll gesture: {confirmation_percent}%", 
            (10, 310), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, 
            (255, 255, 0), 
            2
        )
        
        # Check if gesture has been held long enough to confirm
        if state.gesture_confirmation_counter >= confirmation_frames:
            state.confirmed_gesture = True
            
            # Luôn đặt lại điểm tham chiếu khi xác nhận một cử chỉ mới
            # Initialize reference position once gesture is confirmed
            index_tip = hand_landmarks.landmark[8]  # INDEX_FINGER_TIP
            thumb_tip = hand_landmarks.landmark[4]  # THUMB_TIP
            reference_x = (index_tip.x + thumb_tip.x) / 2 * image_width
            reference_y = (index_tip.y + thumb_tip.y) / 2 * image_height
            
            # Set the reference position that will be used to calculate scroll distance
            state.prev_x = reference_x
            state.prev_y = reference_y
            
            # Draw the reference point to show it's been set
            cv2.circle(
                image,
                (int(reference_x), int(reference_y)),
                8,
                (0, 255, 0),  # Green color
                -1  # Filled circle
            )
            
            cv2.putText(
                image, 
                "Scroll Gesture Confirmed! Reference point set.", 
                (10, 350), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                (0, 255, 0), 
                2
            )
    else:
        # Reset confirmation counter if gesture is not valid
        state.gesture_confirmation_counter = 0
    
    # Update state
    scroll_state = state
    return image

def reset_scroll_orientation():
    """Reset the scroll orientation to allow switching directions"""
    global scroll_state
    if scroll_state.confirmed_gesture:
        scroll_state.scroll_orientation = None