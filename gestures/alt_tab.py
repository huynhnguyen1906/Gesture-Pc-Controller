#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Alt+Tab gesture module to control Alt+Tab feature with gestures
- Open 5 fingers and move up (using DEFAULT_Y_MOVEMENT_THRESHOLD from config) → activates Alt+Tab and holds Alt
- "OK" gesture (index finger and thumb forming a circle) → press Space key and release Alt
"""

import cv2
import numpy as np
from pynput.keyboard import Key, Controller as KeyboardController
from config import keyboard, DEFAULT_Y_MOVEMENT_THRESHOLD
from gestures.base import GestureState, is_open_hand, is_alt_tab_ok_gesture
from config import mouse
from pynput.mouse import Button

# Create state for Alt+Tab gesture
alt_tab_state = GestureState()

def process_alt_tab_gesture(image, hand_landmarks, fps, image_width, image_height):
    """Process open hand gesture with upward movement to activate Alt+Tab and hold Alt key"""
    global alt_tab_state
    state = alt_tab_state
    
    # Calculate frames needed for gesture confirmation and cooldown
    confirmation_frames = int(state.gesture_confirmation_time * fps)
    
    # First check: if we're in Alt state and OK gesture is detected, release Alt and press Space
    if state.is_alt_pressed and is_alt_tab_ok_gesture(hand_landmarks):
        # Release Alt key
        keyboard.release(Key.alt)
        state.is_alt_pressed = False
        state.alt_tab_activated = False
        
        # Press Space key instead of mouse click
        keyboard.press(Key.space)
        keyboard.release(Key.space)
        
        # Display completion message
        cv2.putText(
            image, 
            "Alt + Space successful!", 
            (10, 230), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.9, 
            (0, 255, 255),  # Yellow
            2
        )
        
        # Reset state
        state.confirmed_gesture = False
        state.gesture_confirmation_counter = 0
        state.prev_y = None
        state.cooldown_counter = int(0.5 * fps)  # 0.5 second cooldown
        
        # Update state and return
        alt_tab_state = state
        return image
    
    # If Alt is already being held, display the status but don't return early
    # This allows other gestures to be processed while Alt is held
    if state.is_alt_pressed:
        cv2.putText(
            image, 
            "Holding ALT - Make OK gesture to press Space and release", 
            (10, 230), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, 
            (0, 255, 0),  # Green
            2
        )
        
        # Update state but don't return - allow processing to continue
        alt_tab_state = state
        
        # Only proceed with Alt+Tab gesture processing if we're detecting the OK gesture
        # Otherwise let other gesture detection continue
        if not is_alt_tab_ok_gesture(hand_landmarks):
            return image
    
    # Normal open hand gesture processing
    # Check if we're in cooldown period
    if state.cooldown_counter > 0:
        # Display cooldown message
        cv2.putText(
            image, 
            f"Alt+Tab cooldown: {state.cooldown_counter / fps:.1f}s", 
            (10, 230), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, 
            (255, 165, 0),  # Orange
            2
        )
        
        # Update state and return early
        alt_tab_state = state
        return image
        
    # Check if gesture is still valid (open hand)
    is_valid_gesture = is_open_hand(hand_landmarks)
    
    # If gesture was confirmed but is no longer valid, cancel it
    if state.confirmed_gesture and not is_valid_gesture:
        state.confirmed_gesture = False
        state.gesture_confirmation_counter = 0
        state.prev_y = None
        
        # Display cancellation message
        cv2.putText(
            image, 
            "Alt+Tab canceled: Invalid gesture", 
            (10, 230), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, 
            (0, 0, 255),  # Red
            2
        )
        
        # Update state and return
        alt_tab_state = state
        return image
    
    # If gesture was confirmed, track movement
    if state.confirmed_gesture:
        # Get wrist position for tracking movement
        wrist = hand_landmarks.landmark[0]  # WRIST
        current_y = wrist.y * image_height
        
        # Display active gesture status
        cv2.putText(
            image, 
            "Alt+Tab Gesture Active", 
            (10, 230), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, 
            (0, 255, 0),  # Green
            2
        )
        
        # Only track movement if we have a previous position
        if state.prev_y is not None:
            y_movement = state.prev_y - current_y  # Positive if moving up
            
            # Display movement direction and magnitude
            cv2.putText(
                image, 
                f"Y Movement: {y_movement:.1f}", 
                (10, 270), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                (255, 0, 0),  # Blue 
                2
            )
            
            # Determine gesture direction for significant upward movement
            if y_movement > state.y_movement_threshold and not state.alt_tab_activated:
                # Display Alt+Tab activation
                cv2.putText(
                    image, 
                    "ALT + TAB Activated!", 
                    (10, 310), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.9, 
                    (0, 0, 255),  # Red
                    2
                )
                
                # Press Alt key and Tab key (Alt is held)
                keyboard.press(Key.alt)
                keyboard.press(Key.tab)
                keyboard.release(Key.tab)
                
                # Mark as Alt is being held
                state.is_alt_pressed = True
                state.alt_tab_activated = True
                
                # Reset the gesture detection process but continue holding Alt
                state.confirmed_gesture = False
                state.gesture_confirmation_counter = 0
                state.prev_y = None
        
        # Update previous position
        state.prev_y = current_y
        state.gesture_active = True
        
    # If gesture is not yet confirmed but is valid, increment confirmation counter
    elif is_valid_gesture:
        # Increment confirmation counter
        state.gesture_confirmation_counter += 1
        
        # Display confirmation progress
        confirmation_percent = min(100, int((state.gesture_confirmation_counter / confirmation_frames) * 100))
        cv2.putText(
            image, 
            f"Confirming Alt+Tab gesture: {confirmation_percent}%", 
            (10, 230), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, 
            (255, 255, 0),  # Yellow
            2
        )
        
        # Check if gesture has been held long enough to confirm
        if state.gesture_confirmation_counter >= confirmation_frames:
            state.confirmed_gesture = True
            # Get initial wrist position once gesture is confirmed
            wrist = hand_landmarks.landmark[0]  # WRIST
            state.prev_y = wrist.y * image_height
            
            cv2.putText(
                image, 
                "Alt+Tab Gesture Confirmed!", 
                (10, 270), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                (0, 255, 0),  # Green
                2
            )
    else:
        # Reset confirmation counter if gesture is not valid
        state.gesture_confirmation_counter = 0
    
    # Update state
    alt_tab_state = state
    return image
