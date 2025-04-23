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

# Initialize variables for tracking movement
prev_x = None
gesture_active = False
movement_threshold = 30  # Minimum X movement to trigger a keystroke
cooldown_counter = 0
cooldown_frames = 10  # Wait this many frames after a key press

# Add gesture confirmation variables
gesture_confirmation_time = 0.1  # Seconds to confirm gesture before tracking (reduced to 1/3 of 0.5s)
gesture_confirmation_counter = 0
gesture_cooldown_time = 0.15  # Seconds between gesture detections (reduced to 1/2 of 0.5s)
gesture_cooldown_counter = 0
last_gesture_time = 0
confirmed_gesture = False

def get_available_cameras():
    """
    Detect available cameras in the system
    Returns a list of available cameras with their corresponding indexes
    """
    available_cameras = []
    max_cameras_to_check = 10  # Limit the number of cameras to check
    
    print("Searching for available cameras...")
    
    for i in range(max_cameras_to_check):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)  # Using CAP_DSHOW to speed up connection on Windows
        if cap.isOpened():
            # Read a frame to verify the camera is working
            ret, frame = cap.read()
            if ret:
                # Get camera information if possible
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                available_cameras.append({
                    'index': i,
                    'width': width,
                    'height': height,
                    'fps': fps,
                    'name': f"Camera {i}"
                })
            cap.release()
    
    return available_cameras

def select_camera():
    """
    Display list of available cameras and allow user to select one
    Returns information about the selected camera
    """
    cameras = get_available_cameras()
    
    if not cameras:
        print("No cameras found in the system!")
        return {'index': 0, 'width': 640, 'height': 480, 'fps': 30}
    
    print("\n=== AVAILABLE CAMERAS ===")
    for i, camera in enumerate(cameras):
        print(f"{i+1}. {camera['name']} ({camera['width']}x{camera['height']}, {camera['fps']} FPS)")
    
    choice = -1
    while choice < 0 or choice >= len(cameras):
        try:
            choice = int(input(f"\nSelect camera (1-{len(cameras)}): ")) - 1
            if choice < 0 or choice >= len(cameras):
                print(f"Please choose a number from 1 to {len(cameras)}")
        except ValueError:
            print("Please enter a valid number")
    
    selected_camera = cameras[choice]
    print(f"Selected: {selected_camera['name']} ({selected_camera['width']}x{selected_camera['height']}, {selected_camera['fps']} FPS)")
    
    # Ask user if they want to increase resolution
    try:
        use_high_res = input("Do you want to use higher resolution? (y/n): ").lower() == 'y'
        if use_high_res:
            res_options = [
                (640, 480),
                (800, 600),
                (1280, 720),
                (1920, 1080),
                (2560, 1440),
                (3840, 2160)
            ]
            
            print("\nAvailable resolutions:")
            for i, res in enumerate(res_options):
                print(f"{i+1}. {res[0]}x{res[1]}")
                
            res_choice = -1
            while res_choice < 0 or res_choice >= len(res_options):
                try:
                    res_choice = int(input(f"Select resolution (1-{len(res_options)}): ")) - 1
                    if res_choice < 0 or res_choice >= len(res_options):
                        print(f"Please choose a number from 1 to {len(res_options)}")
                except ValueError:
                    print("Please enter a valid number")
            
            selected_camera['width'] = res_options[res_choice][0]
            selected_camera['height'] = res_options[res_choice][1]
    except Exception as e:
        print(f"Error setting resolution: {e}")
        print("Using default resolution.")
    
    return selected_camera

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
    
    # Select camera before starting
    selected_camera = select_camera()
    camera_index = selected_camera['index']
    width = selected_camera['width']
    height = selected_camera['height']
    
    # Initialize webcam with selected camera
    print(f"Initializing camera with index {camera_index}...")
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)  # Add CAP_DSHOW to speed up initialization
    
    # Configure camera for best performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    
    # Increase frame rate to maximum
    cap.set(cv2.CAP_PROP_FPS, 60)  # Request 60 FPS (if camera supports it)
    
    # Reduce camera buffer (to decrease latency)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    if not cap.isOpened():
        print(f"Error: Cannot open camera with index {camera_index}.")
        return
    
    # Get actual camera parameters after configuration
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = int(cap.get(cv2.CAP_PROP_FPS))
    
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