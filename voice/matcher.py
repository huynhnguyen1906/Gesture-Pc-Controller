#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Text matching module using rapidfuzz
Matches transcribed text with predefined voice commands
"""

from rapidfuzz import fuzz, process
from typing import Optional, Tuple, List
from voice.commands import get_all_commands, get_command_info

class CommandMatcher:
    def __init__(self, min_confidence: float = 0.6):
        """
        Initialize command matcher
        
        Args:
            min_confidence: Minimum confidence score for command matching
        """
        self.min_confidence = min_confidence
        
    def find_best_match(self, transcribed_text: str) -> Optional[Tuple[str, float]]:
        """
        Find the best matching command for transcribed text
        
        Args:
            transcribed_text: Text from speech recognition
            
        Returns:
            Tuple of (command_key, confidence_score) or None if no good match
        """
        if not transcribed_text or not transcribed_text.strip():
            return None
            
        # Get all available commands
        available_commands = get_all_commands()
        
        if not available_commands:
            print("No commands available for matching!")
            return None
        
        # Clean the input text
        cleaned_text = transcribed_text.strip()
        print(f"Matching text: '{cleaned_text}'")
        
        # Find best match using rapidfuzz
        try:
            # Use different matching algorithms and take the best score
            ratio_scores = []
            partial_scores = []
            token_scores = []
            
            for command in available_commands:
                ratio_score = fuzz.ratio(cleaned_text, command)
                partial_score = fuzz.partial_ratio(cleaned_text, command)
                token_score = fuzz.token_ratio(cleaned_text, command)
                
                ratio_scores.append((command, ratio_score))
                partial_scores.append((command, partial_score))
                token_scores.append((command, token_score))
            
            # Get best scores from each method
            best_ratio = max(ratio_scores, key=lambda x: x[1])
            best_partial = max(partial_scores, key=lambda x: x[1])
            best_token = max(token_scores, key=lambda x: x[1])
            
            # Choose the best overall match
            candidates = [best_ratio, best_partial, best_token]
            best_match = max(candidates, key=lambda x: x[1])
            
            command_key, confidence = best_match
            confidence_normalized = confidence / 100.0  # Convert to 0-1 range
            
            print(f"Best match: '{command_key}' with confidence: {confidence_normalized:.2f}")
            
            # Check if confidence meets minimum threshold
            command_info = get_command_info(command_key)
            if command_info:
                command_threshold = command_info.get('confidence_threshold', self.min_confidence)
            else:
                command_threshold = self.min_confidence
                
            if confidence_normalized >= command_threshold:
                return (command_key, confidence_normalized)
            else:
                print(f"Confidence {confidence_normalized:.2f} below threshold {command_threshold:.2f}")
                return None
                
        except Exception as e:
            print(f"Error during text matching: {e}")
            return None
    
    def get_all_matches(self, transcribed_text: str, limit: int = 3) -> List[Tuple[str, float]]:
        """
        Get top N matches for transcribed text
        
        Args:
            transcribed_text: Text from speech recognition
            limit: Maximum number of matches to return
            
        Returns:
            List of (command_key, confidence_score) tuples
        """
        if not transcribed_text or not transcribed_text.strip():
            return []
            
        available_commands = get_all_commands()
        cleaned_text = transcribed_text.strip()
        
        try:
            # Get matches using process.extract
            matches = process.extract(
                cleaned_text, 
                available_commands, 
                scorer=fuzz.ratio,
                limit=limit
            )
            
            # Convert to our format and normalize scores
            result = []
            for command, score, _ in matches:
                confidence = score / 100.0
                if confidence >= self.min_confidence:
                    result.append((command, confidence))
            
            return result
            
        except Exception as e:
            print(f"Error getting all matches: {e}")
            return []
    
    def set_confidence_threshold(self, threshold: float):
        """Set minimum confidence threshold"""
        self.min_confidence = max(0.0, min(1.0, threshold))

# Global matcher instance
matcher = CommandMatcher()
