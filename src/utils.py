import cv2
import numpy as np
from collections import deque

frame_stack = deque(maxlen=4)

def preprocess(frame):

    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    resized = cv2.resize(gray, (84, 84))

    normalized = resized / 255.0

    return normalized.astype(np.float32)

def reset_stack(frame):

    processed = preprocess(frame)

    frame_stack.clear()

    for _ in range(4):
        frame_stack.append(processed)

    return np.stack(frame_stack, axis=0)

def stack_frames(frame):

    processed = preprocess(frame)

    frame_stack.append(processed)

    return np.stack(frame_stack, axis=0)