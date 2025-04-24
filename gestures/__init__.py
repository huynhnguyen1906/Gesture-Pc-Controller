#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Gesture detection package for PC Controller
"""

# Import from base module
from gestures.base import (
    GestureState,
    is_navigation_gesture,
    is_open_hand,
    is_closed_hand,
    is_index_finger_only,
    is_ok_gesture,
    is_scroll_gesture,  # Add the new scroll gesture detection
    update_all_cooldowns,
    reset_all_gesture_states
)

# Import from gesture modules
from gestures.navigation import (
    navigation_state,
    process_navigation_gesture
)

from gestures.alt_f4 import (
    alt_f4_state,
    process_alt_f4_gesture
)

from gestures.mouse_control import (
    mouse_state,
    process_mouse_control_gesture
)

from gestures.mouse_click import (
    mouse_click_state,
    process_mouse_click_gesture
)

# Import the new scroll module
from gestures.scroll import (
    scroll_state,
    process_scroll_gesture,
    reset_scroll_orientation
)

# Dictionary of all gesture states
gesture_states = {
    'navigation': navigation_state,
    'alt_f4': alt_f4_state,
    'mouse': mouse_state,
    'mouse_click': mouse_click_state,
    'scroll': scroll_state  # Add the new scroll state
}

# All available gesture check functions
gesture_checks = {
    'navigation': is_navigation_gesture,
    'alt_f4_open': is_open_hand,
    'alt_f4_close': is_closed_hand,
    'mouse': is_index_finger_only,
    'mouse_click': is_ok_gesture,
    'scroll': is_scroll_gesture  # Add the new scroll gesture check
}

# All available gesture process functions
gesture_processes = {
    'navigation': process_navigation_gesture,
    'alt_f4': process_alt_f4_gesture,
    'mouse': process_mouse_control_gesture,
    'mouse_click': process_mouse_click_gesture,
    'scroll': process_scroll_gesture  # Add the new scroll process
}