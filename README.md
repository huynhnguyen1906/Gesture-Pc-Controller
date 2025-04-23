# Gesture PC Controller

A computer vision application that allows you to control your PC with hand gestures captured via webcam.

## Features

- Detect hand gestures using webcam
- Control left/right keyboard inputs using hand movements
- Visualize hand landmarks and tracking in real-time

## Requirements

- Python 3.x
- OpenCV
- MediaPipe
- PyInput
- NumPy

## Installation

1. Clone this repository
2. Install requirements:
   ```
   pip install -r requirements.txt
   ```

## Usage

Run the main application:

```
python main.py
```

## How it works

The application uses MediaPipe to detect hand landmarks and then interprets specific gestures to trigger keyboard inputs.
