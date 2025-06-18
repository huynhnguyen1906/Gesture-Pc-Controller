#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
YouTube voice commands
Japanese voice commands for YouTube operations
"""

from pynput.keyboard import Key, KeyCode
from config import keyboard
import time

# Dictionary mapping Japanese voice commands to YouTube actions
YOUTUBE_COMMANDS = {
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

def execute_open_youtube():
    """Execute complex YouTube opening sequence"""
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
