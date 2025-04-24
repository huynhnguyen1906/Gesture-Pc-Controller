#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Gesture PC Controller - Ver 0.5
Uses webcam to detect hand gestures for computer control:
- Navigation gesture: Index and middle fingers extended for left/right navigation
- Alt+F4 gesture: Open hand followed by closed hand to close the active window
- Mouse control: Index finger only extended to control mouse cursor
- Mouse click: OK gesture (thumb and index finger form a circle) to left click
- Scroll: Thumb and index finger in V/L shape for vertical scrolling
Supports both left and right hands and simultaneous dual-hand gestures
"""

import cv2
import numpy as np
from camera import select_camera, initialize_camera

# Import from config
from config import (
    mp_hands, mp_drawing, mp_drawing_styles
)

# Import from gesture package
from gestures import (
    is_navigation_gesture, is_open_hand, is_closed_hand, is_index_finger_only, is_ok_gesture,
    is_scroll_gesture, process_navigation_gesture, process_alt_f4_gesture, 
    process_mouse_control_gesture, process_mouse_click_gesture, process_scroll_gesture,
    gesture_states, update_all_cooldowns, reset_all_gesture_states
)

def main():
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
    
    # Configure hand detection parameters
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,  # Detect up to 2 hands
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5
    )
    
    # Calculate time to measure FPS
    prev_frame_time = 0
    new_frame_time = 0
    
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
        
        # Display FPS and instructions
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
            "Gesture Controls:", 
            (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, 
            (255, 255, 255), 
            2
        )
        
        # Update cooldowns regardless of whether a hand is detected
        update_all_cooldowns(gesture_states, actual_fps)
        
        # Process hand landmarks if detected
        if results.multi_hand_landmarks:
            # Keep track of whether specific gestures have been recognized
            mouse_control_active = False
            mouse_click_active = False
            navigation_active = False
            alt_f4_active = False
            scroll_active = False  # New flag for scroll gesture
            
            # First loop: Process mouse control gesture only (primary hand)
            for hand_landmarks in results.multi_hand_landmarks:
                # Check for mouse control gesture (index finger only)
                if is_index_finger_only(hand_landmarks):
                    process_mouse_control_gesture(image, hand_landmarks, actual_fps, image_width, image_height)
                    mouse_control_active = True
                    break  # Once mouse control is found, exit this loop
            
            # Second loop: Only process click gesture if mouse control is active
            if mouse_control_active and len(results.multi_hand_landmarks) > 1:
                # Process the other hands for potential click gesture
                for hand_landmarks in results.multi_hand_landmarks:
                    # Skip if this is the hand already being used for mouse control
                    if is_index_finger_only(hand_landmarks):
                        continue
                    
                    # Check for OK gesture for mouse click
                    if is_ok_gesture(hand_landmarks):
                        process_mouse_click_gesture(image, hand_landmarks, actual_fps, image_width, image_height)
                        mouse_click_active = True
                        break
            
            # Third loop: Process other gestures if mouse control is not active
            if not mouse_control_active:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Check for scroll gesture (L/V shape with thumb and index)
                    if is_scroll_gesture(hand_landmarks):
                        process_scroll_gesture(image, hand_landmarks, actual_fps, image_width, image_height)
                        scroll_active = True
                        break
                    
                    # Check for navigation gesture (index and middle fingers extended)
                    if not scroll_active and is_navigation_gesture(hand_landmarks):
                        process_navigation_gesture(image, hand_landmarks, actual_fps, image_width, image_height)
                        navigation_active = True
                        break
                    
                    # Check for Alt+F4 gesture (open hand to closed hand)
                    if not (scroll_active or navigation_active) and (is_open_hand(hand_landmarks) or is_closed_hand(hand_landmarks)):
                        process_alt_f4_gesture(image, hand_landmarks, actual_fps, image_width, image_height)
                        alt_f4_active = True
                        break
            
            # Process each hand separately to add labels
            for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Draw hand landmarks on the image
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
                
                # Add hand number to display
                hand_text = f"Hand {i+1}"
                wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                wrist_x = int(wrist.x * image_width)
                wrist_y = int(wrist.y * image_height)
                cv2.putText(
                    image,
                    hand_text,
                    (wrist_x, wrist_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 0),
                    1
                )
            
            # If no recognizable gesture was found
            if not (mouse_control_active or navigation_active or alt_f4_active or mouse_click_active or scroll_active):
                # Reset all gesture states when no valid gesture is detected
                reset_all_gesture_states(gesture_states)
                
                cv2.putText(
                    image, 
                    "No valid gesture detected", 
                    (10, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, 
                    (0, 0, 255), 
                    2
                )
        else:
            # No hands detected, reset all gesture states
            reset_all_gesture_states(gesture_states)
            
            cv2.putText(
                image, 
                "No hand detected", 
                (10, 70), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                (0, 0, 255), 
                2
            )
        
        # Add gesture instruction text
        cv2.putText(
            image, 
            "Navigation: 2 fingers extended → LEFT/RIGHT", 
            (10, image_height - 150), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.5, 
            (255, 255, 255), 
            1
        )
        
        cv2.putText(
            image, 
            "Close window: Open hand → Closed hand = Alt+F4", 
            (10, image_height - 120), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.5, 
            (255, 255, 255), 
            1
        )
        
        cv2.putText(
            image, 
            "Mouse control: Index finger only extended", 
            (10, image_height - 90), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.5, 
            (255, 255, 255), 
            1
        )
        
        cv2.putText(
            image, 
            "Mouse click (with 2nd hand): OK gesture while controlling mouse", 
            (10, image_height - 60), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.5, 
            (255, 255, 255), 
            1
        )
        
        # New instruction for scroll gesture
        cv2.putText(
            image, 
            "Scroll: Thumb + Index in L/V shape → VERTICAL scroll from reference point", 
            (10, image_height - 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.5, 
            (255, 255, 255), 
            1
        )
        
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