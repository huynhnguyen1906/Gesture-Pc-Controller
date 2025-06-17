#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Voice command definitions for browser control
Japanese voice commands with their corresponding actions
"""

from pynput.keyboard import Key, KeyCode
from config import keyboard

# Dictionary mapping Japanese voice commands to actions
VOICE_COMMANDS = {
    "ユチュブを開いて": {
        "description": "Open YouTube",
        "action": "open_youtube", 
        "keys": None,  
        "confidence_threshold": 0.6  
    },
    "YouTube開いて": {
        "description": "Open YouTube",
        "action": "open_youtube",
        "keys": None,
        "confidence_threshold": 0.6
    },
    "YouTubeを開いて": {
        "description": "Open YouTube", 
        "action": "open_youtube",
        "keys": None,
        "confidence_threshold": 0.6
    },
    "youtube開いて": {
        "description": "Open YouTube",
        "action": "open_youtube", 
        "keys": None,
        "confidence_threshold": 0.6
    }
}

def execute_command(command_key):
    """Execute the action for a given command key"""
    if command_key not in VOICE_COMMANDS:
        print(f"Unknown command: {command_key}")
        return False
    
    command = VOICE_COMMANDS[command_key]
    action = command["action"]
    
    print(f"Executing: {command['description']}")
    
    try:
        # Handle complex actions
        if action == "open_youtube":
            return execute_open_youtube()
        
        print(f"Command executed successfully: {command['description']}")
        return True
        
    except Exception as e:
        print(f"Error executing command: {e}")
        return False

def execute_open_youtube():
    """Execute complex YouTube opening sequence"""
    import time
    
    try:
        # Step 1: Ctrl+T (new tab)
        print("Step 1: Opening new tab...")
        keyboard.press(Key.ctrl)
        keyboard.press(KeyCode.from_char('t'))
        keyboard.release(KeyCode.from_char('t'))
        keyboard.release(Key.ctrl)
        
        # Wait a moment for tab to open
        time.sleep(0.5)
        
        # Step 2: Type youtube.com
        print("Step 2: Typing youtube.com...")
        keyboard.type("youtube.com")
        
        # Wait a moment before pressing Enter
        time.sleep(0.3)
        
        # Step 3: Press Enter
        print("Step 3: Pressing Enter...")
        keyboard.press(Key.enter)
        keyboard.release(Key.enter)
        
        print("YouTube opened successfully!")
        return True
        
    except Exception as e:
        print(f"Error opening YouTube: {e}")
        return False

def get_all_commands():
    """Get list of all available commands"""
    return list(VOICE_COMMANDS.keys())

def get_command_info(command_key):
    """Get information about a specific command"""
    return VOICE_COMMANDS.get(command_key, None)
