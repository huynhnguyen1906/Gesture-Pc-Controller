#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Gesture PC Controller - Ver 0.2
Uses webcam to detect hand gestures and control keyboard (left/right keys)
Supports both left and right hands
"""

import cv2
import mediapipe as mp
import numpy as np
from pynput.keyboard import Key, Controller
from utils import (
    prev_x, gesture_active, movement_threshold, cooldown_counter, cooldown_frames,
    gesture_confirmation_time, gesture_confirmation_counter, gesture_cooldown_time,
    gesture_cooldown_counter, last_gesture_time, confirmed_gesture
)
from camera import  select_camera, initialize_camera

# Initialize MediaPipe hands module
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Initialize keyboard controller
keyboard = Controller()

# Configure hand detection parameters
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,  # Detect up to 2 hands
    min_detection_confidence=0.6,
    min_tracking_confidence=0.5
)

def is_correct_gesture(hand_landmarks):
    """
    Check if the hand is making the correct gesture:
    - Index and middle fingers extended
    - Ring and pinky fingers bent
    """
    # Get fingertips and knuckles
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    index_pip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    middle_pip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    ring_pip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    pinky_pip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP]
    
    # Check if index and middle fingers are extended (tip y < pip y)
    index_extended = index_tip.y < index_pip.y
    middle_extended = middle_tip.y < middle_pip.y
    
    # Check if ring and pinky fingers are bent (tip y > pip y)
    ring_bent = ring_tip.y > ring_pip.y
    pinky_bent = pinky_tip.y > pinky_pip.y
    
    return index_extended and middle_extended and ring_bent and pinky_bent

def main():
    global prev_x, gesture_active, cooldown_counter
    global gesture_confirmation_counter, gesture_cooldown_counter, confirmed_gesture
    
    # Replace the camera initialization logic
    selected_camera = select_camera()
    camera_index = selected_camera['index']
    width = selected_camera['width']
    height = selected_camera['height']

    try:
        cap, actual_width, actual_height, actual_fps = initialize_camera(camera_index, width, height)
    except ValueError as e:
        print(e)
        return
    
    print(f"Actual camera parameters: {actual_width}x{actual_height} @ {actual_fps} FPS")
    print("Gesture Control Started. Press 'q' to quit.")
    
    # Calculate time to measure FPS
    prev_frame_time = 0
    new_frame_time = 0
    
    # Calculate frames needed for gesture confirmation and cooldown
    confirmation_frames = int(gesture_confirmation_time * actual_fps)
    cooldown_frames = int(gesture_cooldown_time * actual_fps)
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Could not read image from camera.")
            break
        
        # Calculate and display FPS
        new_frame_time = cv2.getTickCount() / cv2.getTickFrequency()
        fps = 1 / (new_frame_time - prev_frame_time) if prev_frame_time > 0 else 0
        prev_frame_time = new_frame_time
        
        # Flip the image horizontally for a more intuitive mirror view
        image = cv2.flip(image, 1)
        image_height, image_width, _ = image.shape
        
        # Convert BGR image to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process the image and detect hands
        results = hands.process(rgb_image)
        
        # Display FPS and other information
        cv2.putText(
            image,
            f"FPS: {fps:.1f}",
            (image_width - 120, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )
        
        cv2.putText(
            image, 
            "Gesture Control: 2 fingers extended, 2 bent", 
            (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, 
            (0, 255, 0), 
            2
        )
        
        # Handle cooldown counters
        if cooldown_counter > 0:
            cooldown_counter -= 1
            
        if gesture_cooldown_counter > 0:
            gesture_cooldown_counter -= 1
            # Display cooldown message
            cv2.putText(
                image, 
                f"Gesture cooldown: {gesture_cooldown_counter / actual_fps:.1f}s", 
                (10, 190), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                (255, 165, 0), 
                2
            )
        
        # Process hand landmarks if detected and not in cooldown
        correct_gesture_detected = False
        current_hand_landmarks = None
        
        if results.multi_hand_landmarks and gesture_cooldown_counter == 0:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks on the image
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
                
                # Check if hand is making the correct gesture
                if is_correct_gesture(hand_landmarks):
                    correct_gesture_detected = True
                    current_hand_landmarks = hand_landmarks
                    break
        
        # Handle gesture confirmation process
        if correct_gesture_detected:
            if not confirmed_gesture:
                # Increment confirmation counter if gesture is correct but not yet confirmed
                gesture_confirmation_counter += 1
                
                # Display confirmation progress
                confirmation_percent = min(100, int((gesture_confirmation_counter / confirmation_frames) * 100))
                cv2.putText(
                    image, 
                    f"Confirming gesture: {confirmation_percent}%", 
                    (10, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, 
                    (255, 255, 0), 
                    2
                )
                
                # Check if gesture has been held long enough to confirm
                if gesture_confirmation_counter >= confirmation_frames:
                    confirmed_gesture = True
                    # Get initial position once gesture is confirmed
                    index_tip = current_hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    middle_tip = current_hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                    prev_x = (index_tip.x + middle_tip.x) / 2 * image_width
                    
                    cv2.putText(
                        image, 
                        "Gesture Confirmed!", 
                        (10, 110), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.7, 
                        (0, 255, 0), 
                        2
                    )
            else:
                # Gesture already confirmed, track movement
                index_tip = current_hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                middle_tip = current_hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                current_x = (index_tip.x + middle_tip.x) / 2 * image_width
                
                # Display active gesture status
                cv2.putText(
                    image, 
                    "Gesture Active", 
                    (10, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, 
                    (0, 255, 0), 
                    2
                )
                
                # Only track movement if we have a previous position
                if prev_x is not None and cooldown_counter == 0:
                    x_movement = prev_x - current_x
                    
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
                    if abs(x_movement) > movement_threshold:
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
                            cooldown_counter = int(0.2 * actual_fps)  # Brief cooldown after key press
                            
                            # Reset the gesture detection process and start cooldown
                            confirmed_gesture = False
                            gesture_confirmation_counter = 0
                            gesture_cooldown_counter = cooldown_frames
                            prev_x = None
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
                            cooldown_counter = int(0.2 * actual_fps)  # Brief cooldown after key press
                            
                            # Reset the gesture detection process and start cooldown
                            confirmed_gesture = False
                            gesture_confirmation_counter = 0
                            gesture_cooldown_counter = cooldown_frames
                            prev_x = None
                
                # Update previous position
                prev_x = current_x
                gesture_active = True
        else:
            # Reset confirmation if gesture is not maintained
            if not confirmed_gesture:
                gesture_confirmation_counter = 0
            
            # If gesture was confirmed but now lost, reset after a brief delay
            if confirmed_gesture and gesture_cooldown_counter == 0:
                confirmed_gesture = False
                prev_x = None
            
            # Display "Wrong Gesture" text if not in cooldown
            if gesture_cooldown_counter == 0:
                cv2.putText(
                    image, 
                    "Wrong Gesture", 
                    (10, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, 
                    (0, 0, 255), 
                    2
                )
            
            gesture_active = False
            
        # Show the image with annotations
        cv2.imshow('Gesture PC Controller', image)
        
        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()