#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Voice command definitions - Legacy file for backward compatibility
All commands are now organized in the commands/ subdirectory
"""

# Import everything from the new modular structure  
from .commands import (
    ALL_COMMANDS as VOICE_COMMANDS,
    execute_command,
    get_all_commands, 
    get_command_info
)

# Re-export the main functions for backward compatibility
__all__ = [
    'VOICE_COMMANDS',
    'execute_command', 
    'get_all_commands',
    'get_command_info'
]
