#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Enhanced Alt+Tab gesture module with advanced hand tracking
- Open 5 fingers → activates Alt+Tab and holds Alt
- Track horizontal hand position → auto press Left/Right arrows based on position  
- Distance from reference point determines frequency (1s/0.5s/0.25s intervals)
- "OK" gesture → confirm selection and release Alt
- No other gestures allowed while Alt+Tab is active
"""

import cv2
import numpy as np
import time
from pynput.keyboard import Key, Controller as KeyboardController
from config import keyboard, DEFAULT_Y_MOVEMENT_THRESHOLD
from gestures.base import GestureState, is_open_hand, is_alt_tab_ok_gesture

# Enhanced state for Alt+Tab gesture
class AltTabState(GestureState):
    def __init__(self):
        super().__init__()
        self.is_alt_pressed = False
        self.alt_tab_activated = False
        self.reference_x = None  # Reference point for horizontal tracking
        self.reference_y = None
        self.last_direction = None  # Track last direction ('left'/'right')
        self.last_arrow_press_time = 0  # Track timing for arrow key presses
        self.current_interval = 1.0  # Current press interval (1s/0.5s/0.25s)

alt_tab_state = AltTabState()

def process_alt_tab_gesture(image, hand_landmarks, fps, image_width, image_height):
    """Enhanced Alt+Tab processing with horizontal tracking"""
    global alt_tab_state
    state = alt_tab_state
    
    # Calculate frames needed for gesture confirmation
    confirmation_frames = int(state.gesture_confirmation_time * fps)
    
    # Priority 1: Handle OK gesture to confirm selection
    if state.is_alt_pressed and is_alt_tab_ok_gesture(hand_landmarks):
        return handle_alt_tab_confirmation(image, state)
    
    # Priority 2: If Alt+Tab is active, handle horizontal tracking
    if state.is_alt_pressed and state.alt_tab_activated:
        return handle_horizontal_tracking(image, hand_landmarks, state, image_width, image_height)
    
    # Priority 3: Handle initial Alt+Tab activation
    return handle_alt_tab_activation(image, hand_landmarks, state, fps, confirmation_frames, image_width, image_height)

def handle_alt_tab_confirmation(image, state):
    """Handle OK gesture to confirm Alt+Tab selection"""
    # Release Alt key
    keyboard.release(Key.alt)
    state.is_alt_pressed = False
    state.alt_tab_activated = False
    
    # Press Space key to confirm selection
    keyboard.press(Key.space)
    keyboard.release(Key.space)
      # Reset all state
    state.confirmed_gesture = False
    state.gesture_confirmation_start_time = 0
    state.reference_x = None
    state.reference_y = None
    state.last_direction = None
    state.last_arrow_press_time = 0
    state.start_gesture_cooldown()  # Cooldown after completion
    
    # Display completion message
    cv2.putText(
        image, 
        "Alt+Tab Confirmed!", 
        (10, 230), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        0.9, 
        (0, 255, 0),  # Green
        2
    )
    
    return image

def handle_horizontal_tracking(image, hand_landmarks, state, image_width, image_height):
    """Handle horizontal hand tracking for Left/Right arrow navigation"""
    # Check if gesture is still valid (open hand)
    if not is_open_hand(hand_landmarks):
        # Cancel Alt+Tab if gesture becomes invalid
        return cancel_alt_tab(image, state)
    
    # Get wrist position as reference point for hand center
    wrist = hand_landmarks.landmark[0]  # WRIST
    current_x = wrist.x * image_width
    current_y = wrist.y * image_height
    
    # Draw current hand position
    cv2.circle(image, (int(current_x), int(current_y)), 8, (0, 0, 255), -1)  # Red dot
    
    # Draw reference point
    if state.reference_x is not None:
        cv2.circle(image, (int(state.reference_x), int(state.reference_y)), 12, (0, 255, 0), 2)  # Green circle
        
        # Draw line connecting reference and current position
        cv2.line(image, (int(state.reference_x), int(state.reference_y)), 
                (int(current_x), int(current_y)), (255, 255, 0), 2)  # Yellow line
        
        # Calculate horizontal distance
        x_distance = current_x - state.reference_x
        
        # Determine direction and interval based on distance
        direction = "right" if x_distance > 0 else "left"
        abs_distance = abs(x_distance)
        
        # Determine press interval based on distance
        if abs_distance > 150:  # Far distance
            interval = 0.25
        elif abs_distance > 75:  # Medium distance  
            interval = 0.5
        else:  # Close distance
            interval = 1.0
        
        state.current_interval = interval
        
        # Display tracking info
        cv2.putText(image, f"Alt+Tab Active - Distance: {int(abs_distance)}px", 
                   (10, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(image, f"Direction: {direction.upper()}, Interval: {interval}s", 
                   (10, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Check if enough time has passed and direction is significant
        current_time = time.time()
        if (current_time - state.last_arrow_press_time >= interval and abs_distance > 20):
            
            # Press appropriate arrow key
            if direction == "left":
                keyboard.press(Key.left)
                keyboard.release(Key.left)
                display_text = "← LEFT"
                color = (255, 0, 0)  # Blue
            else:
                keyboard.press(Key.right) 
                keyboard.release(Key.right)
                display_text = "→ RIGHT"
                color = (0, 255, 255)  # Yellow
            
            # Update timing and direction
            state.last_arrow_press_time = current_time
            state.last_direction = direction
            
            # Display arrow press
            cv2.putText(image, display_text, (10, 290), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    else:
        # Set initial reference point
        state.reference_x = current_x
        state.reference_y = current_y
        state.last_arrow_press_time = time.time()
        
        cv2.putText(image, "Alt+Tab Active - Setting reference point", 
                   (10, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    
    return image

def handle_alt_tab_activation(image, hand_landmarks, state, fps, confirmation_frames, image_width, image_height):
    """Handle initial Alt+Tab activation"""
    # Check if we're in cooldown period
    if state.is_cooldown_active():
        current_time = time.time()
        remaining_cooldown = max(0, state.gesture_cooldown_time - (current_time - state.gesture_cooldown_start_time))
        cv2.putText(image, f"Alt+Tab cooldown: {remaining_cooldown:.1f}s", 
                   (10, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
        return image
    
    # Check if gesture is valid (open hand)
    if not is_open_hand(hand_landmarks):
        # Reset if gesture is not valid
        if state.confirmed_gesture:
            state.confirmed_gesture = False
            state.gesture_confirmation_start_time = 0
            
        cv2.putText(image, "Alt+Tab: Open hand gesture", 
                   (10, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return image
      # Process gesture confirmation
    if not state.confirmed_gesture:
        # Start confirmation timer if not started
        if state.gesture_confirmation_start_time == 0:
            state.start_gesture_confirmation()
        
        # Calculate confirmation progress
        current_time = time.time()
        elapsed_time = current_time - state.gesture_confirmation_start_time
        progress = min(1.0, elapsed_time / state.gesture_confirmation_time)
        
        # Show confirmation progress
        cv2.putText(image, f"Alt+Tab confirming... {progress:.1%}", 
                   (10, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Check if gesture is confirmed
        if state.is_gesture_confirmed():
            state.confirmed_gesture = True
            state.gesture_confirmation_start_time = 0
            
            # Activate Alt+Tab
            keyboard.press(Key.alt)
            keyboard.press(Key.tab)
            keyboard.release(Key.tab)
            
            state.is_alt_pressed = True
            state.alt_tab_activated = True
            
            cv2.putText(image, "Alt+Tab Activated!", 
                       (10, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    return image

def cancel_alt_tab(image, state):
    """Cancel Alt+Tab and reset state"""
    if state.is_alt_pressed:
        keyboard.release(Key.alt)
        state.is_alt_pressed = False
     
    state.alt_tab_activated = False
    state.confirmed_gesture = False
    state.gesture_confirmation_start_time = 0
    state.reference_x = None
    state.reference_y = None
    state.last_direction = None
    state.start_gesture_cooldown()  # Short cooldown
    cv2.putText(image, "Alt+Tab Cancelled", 
               (10, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    return image
