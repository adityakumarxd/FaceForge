import cv2
import numpy as np

def extract_features(video_path):
    """
    Extracts features from a video file for DeepFake detection.
    
    Parameters:
        video_path (str): The path to the video file.
        
    Returns:
        features (list): A list of extracted features.
    """
    features = []
    cap = cv2.VideoCapture(video_path)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Example feature extraction: resizing and converting to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized_frame = cv2.resize(gray_frame, (64, 64))
        features.append(resized_frame.flatten())
    
    cap.release()
    return features

def preprocess_video(video_path):
    """
    Preprocesses the video for detection.
    
    Parameters:
        video_path (str): The path to the video file.
        
    Returns:
        processed_video (list): A list of processed video frames.
    """
    processed_video = []
    cap = cv2.VideoCapture(video_path)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        normalized_frame = frame / 255.0
        processed_video.append(normalized_frame)
    
    cap.release()
    return processed_video