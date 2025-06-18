#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Voice commands module
Centralized import and management of all voice commands
"""

from .youtube import YOUTUBE_COMMANDS, execute_open_youtube
from .tab_control import TAB_COMMANDS, execute_close_tab

# Combine all commands into one dictionary
ALL_COMMANDS = {}
ALL_COMMANDS.update(YOUTUBE_COMMANDS)
ALL_COMMANDS.update(TAB_COMMANDS)

def execute_command(command_key):
    """Execute the action for a given command key"""
    if command_key not in ALL_COMMANDS:
        print(f"Unknown command: {command_key}")
        return False
    
    command = ALL_COMMANDS[command_key]
    action = command["action"]
    
    print(f"Executing: {command['description']}")
    
    try:
        # Handle complex actions
        if action == "open_youtube":
            return execute_open_youtube()
        elif action == "close_tab":
            return execute_close_tab()
        
        print(f"Command executed successfully: {command['description']}")
        return True
        
    except Exception as e:
        print(f"Error executing command: {e}")
        return False

def get_all_commands():
    """Get list of all available commands"""
    return list(ALL_COMMANDS.keys())

def get_command_info(command_key):
    """Get information about a specific command"""
    return ALL_COMMANDS.get(command_key, None)
