#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Speech-to-text transcription using OpenAI Whisper
Converts audio to Japanese text for command recognition
"""

import whisper
import numpy as np
import tempfile
import os
from typing import Optional
import soundfile as sf
import torch

class WhisperTranscriber:
    def __init__(self, model_name: str = "small"):
        """
        Initialize Whisper transcriber
        
        Args:
            model_name: Whisper model size (tiny, base, small, medium, large)
        """
        self.model_name = model_name
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._load_model()
    
    def _load_model(self):
        """Load the Whisper model"""
        try:
            # Check GPU availability with detailed info
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                gpu_name = torch.cuda.get_device_name(0)
                print(f"🚀 Loading Whisper model '{self.model_name}' on **GPU ({gpu_name})** ...")
                print(f"   💡 GPU Count: {gpu_count}, CUDA Version: {torch.version.cuda}")
            else:
                print(f"🐢 Loading Whisper model '{self.model_name}' on **CPU** (no GPU available) ...")
                print("   💡 To enable GPU: install PyTorch with CUDA support")
                print("   💡 Command: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
            
            self.model = whisper.load_model(self.model_name).to(self.device)
            print("✅ Whisper model loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading Whisper model: {e}")
            self.model = None
    
    def transcribe_audio(self, audio_data: np.ndarray, sample_rate: int = 16000) -> Optional[str]:
        """
        Transcribe audio data to text
        
        Args:
            audio_data: Audio data as numpy array
            sample_rate: Sample rate of the audio
            
        Returns:
            Transcribed text or None if error
        """
        if self.model is None:
            print("⚠️ Whisper model not loaded!")
            return None
            
        if audio_data is None or len(audio_data) == 0:
            print("⚠️ No audio data to transcribe!")
            return None
        
        try:
            # Create temporary audio file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_filename = temp_file.name
                
            # Flatten audio data if it's 2D
            if len(audio_data.shape) > 1:
                audio_data = audio_data.flatten()
                
            # Save audio to .wav
            sf.write(temp_filename, audio_data, sample_rate)
            
            # Transcribe using Whisper
            print("📝 Transcribing audio...")
            result = self.model.transcribe(
                temp_filename,
                language="ja",  # Japanese
                fp16=(self.device == "cuda"),  # Only use fp16 on GPU
                verbose=False
            )
            
            # Delete temp file
            os.unlink(temp_filename)
            
            transcribed_text = result.get("text", "").strip()
            print(f"📄 Transcribed: '{transcribed_text}'")
            
            return transcribed_text
            
        except Exception as e:
            print(f"❌ Error transcribing audio: {e}")
            try:
                if 'temp_filename' in locals():
                    os.unlink(temp_filename)
            except:
                pass
            return None
    
    def is_model_loaded(self) -> bool:
        """Check if model is loaded and ready"""
        return self.model is not None
    
    def get_device_info(self) -> dict:
        """Get detailed device information"""
        info = {
            "device": self.device,
            "cuda_available": torch.cuda.is_available(),
            "device_count": 0,
            "device_name": "N/A",
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else "N/A"
        }
        
        if torch.cuda.is_available():
            info["device_count"] = torch.cuda.device_count()
            info["device_name"] = torch.cuda.get_device_name(0)
            
        return info

# Global transcriber instance
transcriber = WhisperTranscriber()
