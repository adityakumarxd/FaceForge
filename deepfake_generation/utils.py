import cv2
import numpy as np

def load_video(video_path):
    """Load a video from the specified path."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Error opening video file: {video_path}")
    return cap

def save_frame(frame, output_path):
    """Save a single frame as an image."""
    cv2.imwrite(output_path, frame)

def extract_frames(video_capture, frame_interval=30):
    """Extract frames from a video capture object at specified intervals."""
    frames = []
    count = 0
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        if count % frame_interval == 0:
            frames.append(frame)
        count += 1
    return frames

def preprocess_frame(frame):
    """Preprocess a frame for model input (resize, normalize, etc.)."""
    frame = cv2.resize(frame, (256, 256))  # Example resize
    frame = frame / 255.0  # Normalize to [0, 1]
    return frame

def postprocess_frame(frame):
    """Postprocess a frame after model prediction (denormalize, etc.)."""
    frame = frame * 255.0  # Denormalize to [0, 255]
    return np.clip(frame, 0, 255).astype(np.uint8)  # Ensure valid pixel values