#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tab control voice commands
Japanese voice commands for browser tab operations
"""

from pynput.keyboard import Key, KeyCode
from config import keyboard
import time

# Dictionary mapping Japanese voice commands to tab control actions
TAB_COMMANDS = {
    "タブを閉じて": {
        "description": "Close current tab",
        "action": "close_tab",
        "keys": None,
        "confidence_threshold": 0.6
    },
    "tabを閉じて": {
        "description": "Close current tab",
        "action": "close_tab",
        "keys": None,
        "confidence_threshold": 0.6
    },
    "タブ閉じて": {
        "description": "Close current tab",
        "action": "close_tab",
        "keys": None,
        "confidence_threshold": 0.6
    },
    "tab閉じて": {
        "description": "Close current tab",
        "action": "close_tab",
        "keys": None,
        "confidence_threshold": 0.6
    }
}

def execute_close_tab():
    """Execute close tab action using Ctrl+W"""
    try:
        print("Closing current tab...")
        
        # Press Ctrl+W to close current tab
        keyboard.press(Key.ctrl)
        keyboard.press(KeyCode.from_char('w'))
        keyboard.release(KeyCode.from_char('w'))
        keyboard.release(Key.ctrl)
        
        print("Tab closed successfully!")
        return True
        
    except Exception as e:
        print(f"Error closing tab: {e}")
        return False
