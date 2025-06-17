#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Scroll gesture module for controlling scrolling with thumb and index finger in L/V shape
Only allows vertical scrolling with reference point tracking
"""

import cv2
import time
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
    
    # Check if we're in cooldown period after a successful scroll gesture
    if state.is_cooldown_active():
        current_time = time.time()
        remaining_cooldown = max(0, state.gesture_cooldown_time - (current_time - state.gesture_cooldown_start_time))
        
        # Display cooldown message
        cv2.putText(
            image, 
            f"Scroll cooldown: {remaining_cooldown:.1f}s", 
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
    
    # If gesture was confirmed but is no longer valid, cancel it
    if state.confirmed_gesture and not is_valid_gesture:
        state.confirmed_gesture = False
        state.gesture_confirmation_start_time = 0
        state.prev_y = None
        state.scroll_orientation = None
        
        cv2.putText(
            image, 
            "Scroll gesture canceled", 
            (10, 310), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, 
            (0, 0, 255), 
            2
        )
        
        # Update state and return
        scroll_state = state
        return image

    # Process confirmed gesture
    if state.confirmed_gesture and is_valid_gesture:
        # Get thumb and index finger tips for movement calculation
        thumb_tip = hand_landmarks.landmark[4]  # THUMB_TIP
        index_tip = hand_landmarks.landmark[8]  # INDEX_FINGER_TIP
        
        # Calculate current position between thumb and index finger
        current_center_x = (thumb_tip.x + index_tip.x) / 2 * image_width
        current_center_y = (thumb_tip.y + index_tip.y) / 2 * image_height
        
        # Draw fixed reference point (if exists)
        if hasattr(state, 'reference_x') and hasattr(state, 'reference_y'):
            cv2.circle(image, (int(state.reference_x), int(state.reference_y)), 15, (0, 255, 255), -1)  # Yellow filled circle
            cv2.circle(image, (int(state.reference_x), int(state.reference_y)), 15, (0, 0, 0), 2)  # Black outline
            
            cv2.putText(
                image, 
                "REF", 
                (int(state.reference_x) - 15, int(state.reference_y) + 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                (0, 0, 0), 
                2
            )
        
        # Draw current hand position
        cv2.circle(image, (int(current_center_x), int(current_center_y)), 8, (0, 255, 0), -1)  # Green current position
        
        cv2.putText(
            image, 
            "Scroll Gesture Active - Move vertically from reference point", 
            (10, 310), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, 
            (0, 255, 0), 
            2
        )
        
        # Track movement for scrolling (only vertical, relative to reference point)
        current_time = time.time()
        if (hasattr(state, 'reference_y') and 
            not (current_time - state.cooldown_start_time < state.cooldown_duration)):
            
            y_movement = state.reference_y - current_center_y
            
            # Display movement relative to reference
            cv2.putText(
                image, 
                f"Y Move from ref: {y_movement:.1f}", 
                (10, 350), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                (255, 0, 0), 
                2
            )
            
            # Determine scroll direction and magnitude for vertical movement only
            abs_y_movement = abs(y_movement)
            
            if abs_y_movement > state.movement_threshold:
                # Calculate scroll intensity based on movement magnitude
                scroll_intensity = max(1, min(5, int(abs_y_movement / 15)))
                
                if y_movement > 0:  # Upward movement from reference
                    cv2.putText(
                        image, 
                        f"SCROLL UP (intensity: {scroll_intensity})", 
                        (10, 390), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.9, 
                        (0, 255, 0), 
                        2
                    )
                    # Scroll up
                    mouse.scroll(0, scroll_intensity)
                    
                    # Brief cooldown to prevent too rapid scrolling
                    state.start_cooldown(max(0.05, 0.1 / scroll_intensity))
                    
                elif y_movement < 0:  # Downward movement from reference
                    cv2.putText(
                        image, 
                        f"SCROLL DOWN (intensity: {scroll_intensity})", 
                        (10, 390), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.9, 
                        (0, 255, 0), 
                        2
                    )
                    # Scroll down
                    mouse.scroll(0, -scroll_intensity)
                    
                    # Brief cooldown to prevent too rapid scrolling
                    state.start_cooldown(max(0.05, 0.1 / scroll_intensity))
        
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
            f"Confirming scroll gesture: {confirmation_percent}%", 
            (10, 310), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, 
            (255, 255, 0), 
            2
        )
        
        # Check if gesture has been held long enough to confirm
        if state.is_gesture_confirmed():
            state.confirmed_gesture = True
            # Set fixed reference point once gesture is confirmed
            thumb_tip = hand_landmarks.landmark[4]  # THUMB_TIP
            index_tip = hand_landmarks.landmark[8]  # INDEX_FINGER_TIP
            state.reference_x = (thumb_tip.x + index_tip.x) / 2 * image_width
            state.reference_y = (thumb_tip.y + index_tip.y) / 2 * image_height
            
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
        # Reset confirmation timer if gesture is not valid
        state.gesture_confirmation_start_time = 0
        
        # If we were in the middle of an active gesture, start cooldown
        if state.confirmed_gesture:
            cv2.putText(
                image, 
                "Scroll gesture ended", 
                (10, 310), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                (0, 255, 255), 
                2
            )
            
            # Reset scroll state
            state.reset()
            
            # Start gesture cooldown to prevent immediate re-triggering
            state.start_gesture_cooldown()
    
    # Update state
    scroll_state = state
    return image
