#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Voice control module for browser commands
Integrates recording, transcription, and command execution
"""

from voice.recorder import recorder
from voice.transcriber import transcriber
from voice.matcher import matcher
from voice.commands import execute_command, get_all_commands

# Export functions for gesture modules
def record_voice_command():
    """Record voice command using the recorder"""
    return recorder.record_sync()

def transcribe_audio(audio_data, sample_rate=16000):
    """Transcribe audio data to text"""
    return transcriber.transcribe_audio(audio_data, sample_rate)

def match_command(text):
    """Match transcribed text to commands"""
    return matcher.match_command(text)

# Voice control state
voice_state = {
    'is_processing': False,
    'last_command': None,
    'last_confidence': 0.0
}

def process_voice_command(audio_data, sample_rate=16000):
    """
    Complete pipeline for processing voice command
    
    Args:
        audio_data: Recorded audio data
        sample_rate: Audio sample rate
        
    Returns:
        Boolean indicating success
    """
    global voice_state
    
    if voice_state['is_processing']:
        print("Already processing voice command!")
        return False
    
    voice_state['is_processing'] = True
    
    try:
        # Step 1: Transcribe audio to text
        print("Step 1: Transcribing audio...")
        transcribed_text = transcriber.transcribe_audio(audio_data, sample_rate)
        
        if not transcribed_text:
            print("No text transcribed from audio")
            return False
        
        # Step 2: Match text with commands
        print("Step 2: Matching command...")
        match_result = matcher.find_best_match(transcribed_text)
        
        if not match_result:
            print("No matching command found")
            return False
        
        command_key, confidence = match_result
        
        # Step 3: Execute command
        print(f"Step 3: Executing command '{command_key}' (confidence: {confidence:.2f})")
        success = execute_command(command_key)
        
        if success:
            voice_state['last_command'] = command_key
            voice_state['last_confidence'] = confidence
            print("Voice command executed successfully!")
        
        return success
        
    except Exception as e:
        print(f"Error processing voice command: {e}")
        return False
        
    finally:
        voice_state['is_processing'] = False

def is_voice_system_ready():
    """Check if all voice components are ready"""
    try:
        return (
            transcriber.is_model_loaded() and
            len(get_all_commands()) > 0
        )
    except:
        return False

def get_voice_state():
    """Get current voice processing state"""
    return voice_state.copy()
