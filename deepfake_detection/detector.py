import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def histogram_difference(face1, face2):
    hist1 = cv2.calcHist([face1], [0, 1, 2], None, [8, 8, 8], [0,256,0,256,0,256])
    hist2 = cv2.calcHist([face2], [0, 1, 2], None, [8, 8, 8], [0,256,0,256,0,256])
    cv2.normalize(hist1, hist1)
    cv2.normalize(hist2, hist2)
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

def detect_deepfake(video_path):
    cap = cv2.VideoCapture(video_path)
    prev_face = None
    count = 0
    differences = []

    while count < 10:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        if len(faces) == 0:
            continue
        x, y, w, h = faces[0]
        face_img = frame[y:y+h, x:x+w]
        if prev_face is not None:
            diff = histogram_difference(prev_face, face_img)
            differences.append(diff)
        prev_face = face_img
        count += 1
    cap.release()

    if not differences:
        return "Unknown", 0.0

    avg_diff = np.mean(differences)
    label = "Fake" if avg_diff < 0.7 else "Real"
    confidence = 1 - avg_diff if label == "Fake" else avg_diff
    return label, float(confidence)
