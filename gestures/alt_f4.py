#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Alt+F4 gesture module for closing active windows
"""

import time
import cv2
from pynput.keyboard import Key, KeyCode
from config import keyboard
from gestures.base import GestureState, is_open_hand, is_closed_hand

# Create state for Alt+F4 gesture
alt_f4_state = GestureState()

def process_alt_f4_gesture(image, hand_landmarks, fps, image_width, image_height):
    """Process open hand to closed hand gesture to trigger Alt+F4"""
    global alt_f4_state
    state = alt_f4_state
    
    current_time = time.time()
    is_open = is_open_hand(hand_landmarks)
    is_closed = is_closed_hand(hand_landmarks)
    
    # Calculate frames needed for gesture confirmation
    confirmation_frames = int(state.gesture_confirmation_time * fps)
    
    # Define a partial hand function - neither fully open nor fully closed
    # Check for at least one finger being different than the others
    is_partial_hand = check_partial_hand(hand_landmarks)
    
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
            
            # Execute Alt+F4 - 
            keyboard.press(Key.alt)
            keyboard.press(Key.f4)  
            keyboard.release(Key.f4)
            keyboard.release(Key.alt)
            
            # Reset state
            state.open_hand_confirmed = False
            state.gesture_confirmation_counter = 0
            state.gesture_cooldown_counter = int(state.gesture_cooldown_time * fps)
    
    # If timeout, gesture changed, or no valid gesture detected (neither open nor closed hand)
    elif state.open_hand_confirmed:
        # Check for any of these conditions to cancel:
        # 1. Not an open hand anymore
        # 2. Detected a partial hand (transition state)
        # 3. Timeout condition
        if not is_open or is_partial_hand:
            # Reset state due to change in gesture before completion
            state.open_hand_confirmed = False
            state.gesture_confirmation_counter = 0
            
            cv2.putText(
                image, 
                "Open-close sequence cancelled: Hand position changed", 
                (10, 310), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                (0, 0, 255), 
                2
            )
        # Check for timeout (3 seconds to complete the gesture)
        elif current_time - state.last_hand_state_change_time > 3.0:
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
            if not is_closed:
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
    alt_f4_state = state
    return image

def check_partial_hand(hand_landmarks):
    """
    Check if the hand is in a partial gesture (neither fully open nor fully closed)
    This helps detect transition states when the user is changing hand positions.
    """
    # Get fingertips and pips (middle joints)
    index_tip = hand_landmarks.landmark[8]  # INDEX_FINGER_TIP
    index_pip = hand_landmarks.landmark[6]  # INDEX_FINGER_PIP
    middle_tip = hand_landmarks.landmark[12]  # MIDDLE_FINGER_TIP
    middle_pip = hand_landmarks.landmark[10]  # MIDDLE_FINGER_PIP
    ring_tip = hand_landmarks.landmark[16]  # RING_FINGER_TIP
    ring_pip = hand_landmarks.landmark[14]  # RING_FINGER_PIP
    pinky_tip = hand_landmarks.landmark[20]  # PINKY_TIP
    pinky_pip = hand_landmarks.landmark[18]  # PINKY_PIP
    
    # Count how many fingers are extended and how many are bent
    extended_count = 0
    bent_count = 0
    
    # Check each finger
    if index_tip.y < index_pip.y:
        extended_count += 1
    else:
        bent_count += 1
        
    if middle_tip.y < middle_pip.y:
        extended_count += 1
    else:
        bent_count += 1
        
    if ring_tip.y < ring_pip.y:
        extended_count += 1
    else:
        bent_count += 1
        
    if pinky_tip.y < pinky_pip.y:
        extended_count += 1
    else:
        bent_count += 1
    
    # A partial hand has some fingers extended and some bent
    # We exclude the cases of all extended (open hand) and all bent (closed hand)
    return extended_count > 0 and bent_count > 0