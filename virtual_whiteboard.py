"""
AI-Based Virtual Whiteboard Using Hand Gesture Recognition
==========================================================
Two separate windows:
  "Camera"     = clean live camera, nothing drawn on it
  "Whiteboard" = a clean board, only your writing

Gestures:
    Index finger only ............ WRITE
    Open palm (4 fingers up) ..... ERASE
    Anything else (fist, 2 fingers, etc.) ... PAUSE (lifts the pen)

Keyboard:  c = clear | u = undo | s = save PNG | + / - = pen size | q or ESC = quit
Run:  python virtual_whiteboard.py   (or: python virtual_whiteboard.py --camera 1)
"""

import argparse
import math
import time
from collections import deque
from datetime import datetime

import cv2
import mediapipe as mp
import numpy as np

# ----------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------
CW, CH = 640, 480              # camera size
BW, BH = 800, 600              # whiteboard size
FOOTER_H = 32

# The hand only needs to move inside the inner part of the camera view to
# reach every corner of the board (easier to reach the edges).
MARGIN_X, MARGIN_Y = 0.10, 0.10

ERASER_RADIUS = 40
PEN_SIZES = [4, 6, 10]
INK = (20, 20, 20)             # single pen colour (near black)
WHITE = (255, 255, 255)

SMOOTHING = 0.45               # higher = more responsive, lower = smoother
STABLE_FRAMES = 3              # frames a gesture must hold before it is accepted
LOST_FRAMES = 6                # frames without a hand before the stroke ends
MAX_JUMP = 200                 # px; bigger jumps start a new stroke (no streaks)
UNDO_LIMIT = 30

DRAW, ERASE, IDLE = "WRITE", "ERASE", "PAUSE"


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def fingers_up(lm):
    """[index, middle, ring, pinky] booleans. Compares distances to the wrist,
    so it works at any hand rotation."""
    wrist = lm[0]
    result = []
    for tip, pip in ((8, 6), (12, 10), (16, 14), (20, 18)):
        result.append(dist(lm[tip], wrist) > dist(lm[pip], wrist) * 1.02)
    return result


def classify(lm):
    index, middle, ring, pinky = fingers_up(lm)
    if index and not middle and not ring and not pinky:
        return DRAW
    if index and middle and ring and pinky:
        return ERASE
    return IDLE


def to_board(nx, ny):
    """Map a normalised camera position (0..1) to board pixels."""
    x = (nx - MARGIN_X) / (1 - 2 * MARGIN_X)
    y = (ny - MARGIN_Y) / (1 - 2 * MARGIN_Y)
    x = min(max(x, 0.0), 1.0)
    y = min(max(y, 0.0), 1.0)
    return int(x * (BW - 1)), int(y * (BH - 1))


def save_canvas(canvas):
    name = datetime.now().strftime("whiteboard_%Y%m%d_%H%M%S.png")
    cv2.imwrite(name, canvas)
    print(f"Saved {name}")
    return name


# ----------------------------------------------------------------------------
# Main application
# ----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CW)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CH)
    if not cap.isOpened():
        raise SystemExit("Could not open the camera. Try --camera 1, and close "
                         "other apps that use the webcam.")

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.6,
    )

    canvas = np.full((BH, BW, 3), 255, dtype=np.uint8)   # the whiteboard
    undo_stack = deque(maxlen=UNDO_LIMIT)
    size_idx = 1

    gesture = IDLE
    history = deque(maxlen=STABLE_FRAMES)
    prev_pt = None
    smooth_pt = None
    stroke_active = False
    lost = 0

    fps, last_t = 0.0, time.time()

    cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Whiteboard", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Camera", CW, CH)
    cv2.resizeWindow("Whiteboard", BW, BH + FOOTER_H)
    cv2.moveWindow("Camera", 20, 60)
    cv2.moveWindow("Whiteboard", CW + 50, 60)

    def push_undo():
        undo_stack.append(canvas.copy())

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Camera frame not received.")
            break

        frame = cv2.flip(frame, 1)                       # mirror view
        cam = cv2.resize(frame, (CW, CH))
        result = hands.process(cv2.cvtColor(cam, cv2.COLOR_BGR2RGB))

        cursor = None

        if result.multi_hand_landmarks:
            lost = 0
            hl = result.multi_hand_landmarks[0]
            lm = [(int(p.x * CW), int(p.y * CH)) for p in hl.landmark]

            history.append(classify(lm))
            if len(history) == STABLE_FRAMES and len(set(history)) == 1:
                new_gesture = history[0]
                if new_gesture != gesture:
                    gesture = new_gesture
                    prev_pt = None            # never connect strokes across gestures
                    smooth_pt = None
                    stroke_active = False

            point = hl.landmark[9] if gesture == ERASE else hl.landmark[8]
            raw = to_board(point.x, point.y)     # palm centre / index tip -> board
            if smooth_pt is None:
                smooth_pt = raw
            else:
                smooth_pt = (
                    int(SMOOTHING * raw[0] + (1 - SMOOTHING) * smooth_pt[0]),
                    int(SMOOTHING * raw[1] + (1 - SMOOTHING) * smooth_pt[1]),
                )
            cursor = smooth_pt
        else:
            lost += 1
            if lost >= LOST_FRAMES:
                gesture = IDLE
                history.clear()
                prev_pt = None
                smooth_pt = None
                stroke_active = False

        # ---------------- act on the gesture ----------------
        if cursor is not None and gesture == DRAW:
            if not stroke_active:
                push_undo()
                stroke_active = True
            if prev_pt is None or dist(prev_pt, cursor) > MAX_JUMP:
                prev_pt = cursor
            thickness = PEN_SIZES[size_idx]
            cv2.line(canvas, prev_pt, cursor, INK, thickness, cv2.LINE_AA)
            cv2.circle(canvas, cursor, max(thickness // 2, 1), INK, -1, cv2.LINE_AA)
            prev_pt = cursor

        elif cursor is not None and gesture == ERASE:
            if not stroke_active:
                push_undo()
                stroke_active = True
            cv2.circle(canvas, cursor, ERASER_RADIUS, WHITE, -1)

        # ---------------- window 1: clean camera ----------------
        cv2.imshow("Camera", cam)

        # ---------------- window 2: whiteboard ----------------
        board = canvas.copy()
        if cursor is not None:
            if gesture == DRAW:
                cv2.circle(board, cursor, PEN_SIZES[size_idx] // 2 + 5, (120, 120, 120), 1, cv2.LINE_AA)
            elif gesture == ERASE:
                cv2.circle(board, cursor, ERASER_RADIUS, (120, 120, 120), 1, cv2.LINE_AA)

        footer = np.full((FOOTER_H, BW, 3), 235, dtype=np.uint8)
        now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(now - last_t, 1e-6))
        last_t = now
        cv2.putText(footer, f"Mode: {gesture}   FPS: {fps:4.1f}", (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (50, 50, 50), 1, cv2.LINE_AA)
        cv2.putText(footer, "c clear | u undo | s save | +/- size | q quit", (BW - 400, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (90, 90, 90), 1, cv2.LINE_AA)

        cv2.imshow("Whiteboard", np.vstack([board, footer]))

        # ---------------- keyboard ----------------
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord("c"):
            push_undo()
            canvas[:] = 255
        elif key == ord("u") and undo_stack:
            canvas = undo_stack.pop()
        elif key == ord("s"):
            save_canvas(canvas)
        elif key in (ord("+"), ord("=")):
            size_idx = min(size_idx + 1, len(PEN_SIZES) - 1)
        elif key in (ord("-"), ord("_")):
            size_idx = max(size_idx - 1, 0)

    cap.release()
    hands.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()