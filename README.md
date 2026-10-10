# AI-Based Virtual Whiteboard Using Hand Gesture Recognition

Write in the air with your finger and see it appear on a digital whiteboard.
A webcam tracks your hand with **MediaPipe**, and **OpenCV** displays the camera
and the board in two separate windows. No mouse, pen, or touch screen needed.

## Features
- Real-time hand tracking (21 landmarks) with MediaPipe Hands, running invisibly in the background
- **Clean camera window:** it shows only your live video, with **no dots, lines, or landmarks drawn on your hand** and no text or buttons on top
- **Clean whiteboard window:** a blank board that shows only your writing
- Gesture control: write, erase, pause
- Smoothing and gesture stabilization for clean handwriting
- Undo, clear, pen size, and save as PNG (keyboard shortcuts)

## How it works
1. OpenCV captures the webcam video and mirrors it.
2. MediaPipe detects the hand and finds its 21 landmarks. This happens internally and is **never drawn** on the camera view.
3. The program checks which fingers are up and classifies the gesture.
4. The fingertip position is mapped to the whiteboard, and your strokes are drawn there.

## Gestures
| Gesture | Action |
|---|---|
| Index finger only | Write |
| Open palm (4 fingers up) | Erase |
| Anything else (fist, two fingers) | Pause (lifts the pen) |

## Keyboard shortcuts
`c` clear | `u` undo | `s` save PNG | `+` / `-` pen size | `q` or `Esc` quit

## Installation
Requires Python 3.9 - 3.12 and a webcam.

```bash
git clone https://github.com/YOUR-USERNAME/virtual-whiteboard.git
cd virtual-whiteboard
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac / Linux
pip install -r requirements.txt
python virtual_whiteboard.py
```

If the camera does not open, try `python virtual_whiteboard.py --camera 1`.

## Tech stack
Python, OpenCV, MediaPipe, NumPy

## Tips
Use good lighting, keep your hand 40-80 cm from the camera, and fold the other
fingers down firmly when writing with the index finger.