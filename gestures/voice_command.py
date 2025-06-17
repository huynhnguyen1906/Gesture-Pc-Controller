#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Voice command gesture module
Uses closed hand gesture to trigger voice recording and processing
"""

import cv2
from gestures.base import GestureState, is_closed_hand

# Create state for voice command gesture
voice_command_state = GestureState()

def process_voice_command_gesture(image, hand_landmarks, fps, image_width, image_height):
    """
    Process closed hand gesture to trigger voice command recording
    """
    global voice_command_state
    state = voice_command_state
    
    # Calculate frames needed for gesture confirmation and cooldown
    confirmation_frames = int(state.gesture_confirmation_time * fps)
    cooldown_frames = int(state.gesture_cooldown_time * fps)
    
    # Check if we're in cooldown period
    if state.gesture_cooldown_counter > 0:
        cv2.putText(
            image, 
            f"Voice cooldown: {state.gesture_cooldown_counter / fps:.1f}s", 
            (10, 200), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, 
            (255, 165, 0),  # Orange
            2
        )
        return image
    
    # Check if gesture is still valid (closed hand)
    if not is_closed_hand(hand_landmarks):
        # Reset gesture state if hand is not closed
        if state.confirmed_gesture:
            state.confirmed_gesture = False
            state.gesture_confirmation_counter = 0
            
        cv2.putText(
            image, 
            "Voice trigger: Make closed fist gesture", 
            (10, 200), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, 
            (255, 255, 255), 
            2
        )
        return image
    
    # Gesture is valid, check confirmation
    if not state.confirmed_gesture:
        state.gesture_confirmation_counter += 1
        
        # Show confirmation progress
        progress = state.gesture_confirmation_counter / confirmation_frames
        cv2.putText(
            image, 
            f"Voice trigger confirming... {progress:.1%}", 
            (10, 200), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, 
            (255, 255, 0),  # Yellow
            2
        )
        
        # Check if gesture is confirmed
        if state.gesture_confirmation_counter >= confirmation_frames:
            state.confirmed_gesture = True
            state.gesture_confirmation_counter = 0
            
            # Trigger voice recording
            trigger_voice_recording()
            
            # Set cooldown
            state.gesture_cooldown_counter = cooldown_frames
    else:
        # Gesture already confirmed, show processing state
        cv2.putText(
            image, 
            "Voice command triggered! Processing...", 
            (10, 200), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, 
            (0, 255, 0),  # Green
            2
        )
    
    return image

def trigger_voice_recording():
    """
    Trigger voice recording and processing pipeline
    """
    try:
        # Import voice modules
        from voice import recorder, process_voice_command, is_voice_system_ready
        
        # Check if voice system is ready
        if not is_voice_system_ready():
            print("Voice system not ready! Please ensure Whisper model is loaded.")
            return
        
        print("Voice command gesture detected! Starting recording...")
        
        # Define callback for when recording is complete
        def on_recording_complete(audio_data):
            if audio_data is not None:
                print("Recording complete, processing voice command...")
                # Process the voice command
                success = process_voice_command(audio_data, recorder.sample_rate)
                if not success:
                    print("Voice command processing failed")
            else:
                print("No audio data recorded")
        
        # Start recording
        recorder.start_recording(callback=on_recording_complete)
        
    except Exception as e:
        print(f"Error triggering voice recording: {e}")
