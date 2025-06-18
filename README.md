# Gesture PC Controller

A computer vision application that allows you to control your PC with hand gestures captured via webcam using AI-powered hand tracking.

## Features

### 🖱️ **Mouse Control**

- **Index finger only** → Control mouse cursor
- **OK gesture (with second hand)** → Left click

### ⌨️ **Keyboard Navigation**

- **2 fingers extended (index + middle)** → LEFT/RIGHT arrow keys

### 📜 **Scrolling**

- **Thumb + Index in L/V shape** → Vertical scrolling

### 🔄 **Alt+Tab Navigation**

- **Open hand (5 fingers) + move up** → Activate Alt+Tab and hold Alt key
- **OK gesture while holding Alt** → Press Space and release Alt

### 🎤 **Voice Commands**

- Voice recognition support using AI transcription

## Requirements

- **Python 3.10** (Required)

## Installation

1. Clone this repository
2. Create virtual environment with Python 3.10:
   ```
   py -3.10 -m venv venv
   venv\Scripts\activate
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

Run the main application:

```
python main.py
```

Press **'q'** to quit the application.

## How it works

The application uses **MediaPipe** for real-time hand landmark detection and **computer vision** to interpret specific hand gestures, converting them into keyboard and mouse inputs. It supports **dual-hand gestures** and provides **real-time visual feedback** with gesture status and instructions.
