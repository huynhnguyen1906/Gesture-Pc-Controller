#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Audio recording module using sounddevice
Records audio for voice command recognition
"""

import sounddevice as sd
import numpy as np
import time
import threading
from typing import Optional, Callable

class VoiceRecorder:
    def __init__(self, sample_rate: int = 16000, duration: float = 3.0):
        """
        Initialize voice recorder
        
        Args:
            sample_rate: Audio sample rate (16kHz recommended for Whisper)
            duration: Maximum recording duration in seconds
        """
        self.sample_rate = sample_rate
        self.duration = duration
        self.is_recording = False
        self.audio_data = None
        self.recording_thread = None
        
    def start_recording(self, callback: Optional[Callable] = None):
        """
        Start recording audio
        
        Args:
            callback: Function to call when recording is complete
        """
        if self.is_recording:
            print("Already recording!")
            return False
            
        print(f"Starting voice recording for {self.duration} seconds...")
        self.is_recording = True
        
        # Start recording in a separate thread
        self.recording_thread = threading.Thread(
            target=self._record_audio,
            args=(callback,)
        )
        self.recording_thread.start()
        return True
    
    def stop_recording(self):
        """Stop recording audio"""
        if not self.is_recording:
            return None
            
        self.is_recording = False
        if self.recording_thread:
            self.recording_thread.join()
            
        return self.audio_data
    
    def _record_audio(self, callback: Optional[Callable] = None):
        """Internal method to record audio"""
        try:
            # Record audio
            self.audio_data = sd.rec(
                int(self.duration * self.sample_rate),
                samplerate=self.sample_rate,
                channels=1,
                dtype=np.float32
            )
            
            # Wait for recording to complete
            sd.wait()
            
            print("Recording completed!")
            self.is_recording = False
            
            # Call callback if provided
            if callback:
                callback(self.audio_data)
                
        except Exception as e:
            print(f"Error during recording: {e}")
            self.is_recording = False
            self.audio_data = None
    
    def get_audio_data(self):
        """Get the recorded audio data"""
        return self.audio_data
    
    def is_recording_active(self):
        """Check if recording is currently active"""
        return self.is_recording
    
    def get_available_devices(self):
        """Get list of available audio input devices"""
        try:
            devices = sd.query_devices()
            input_devices = []
            
            for i, device in enumerate(devices):
                if device['max_input_channels'] > 0:
                    input_devices.append({
                        'id': i,
                        'name': device['name'],
                        'sample_rate': device['default_samplerate']
                    })
            
            return input_devices
        except Exception as e:
            print(f"Error querying audio devices: {e}")
            return []

# Global recorder instance
recorder = VoiceRecorder()
