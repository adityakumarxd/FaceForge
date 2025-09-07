import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def generate_deepfake(img_path, video_path, out_path):
    src_img = cv2.imread(img_path)
    src_gray = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
    src_faces = face_cascade.detectMultiScale(src_gray, 1.3, 5)
    if len(src_faces) == 0:
        print("No face found in source image.")
        return
    x1, y1, w1, h1 = src_faces[0]
    src_face = src_img[y1:y1+h1, x1:x1+w1]

    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        tgt_faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        if len(tgt_faces) == 0:
            out.write(frame)
            continue

        x2, y2, w2, h2 = tgt_faces[0]
        src_face_resized = cv2.resize(src_face, (w2, h2))
        mask = 255 * np.ones(src_face_resized.shape, src_face_resized.dtype)
        center = (x2 + w2//2, y2 + h2//2)
        try:
            blended = cv2.seamlessClone(src_face_resized, frame, mask, center, cv2.NORMAL_CLONE)
            out.write(blended)
        except:
            out.write(frame)
    cap.release()
    out.release()



# import cv2
# import numpy as np
# from tensorflow.keras.models import load_model

# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# face_swap_model = load_model('models/pretrained_generator.h5')  # put your model file here

# def preprocess_face(face, img_size=(128, 128)):
#     face = cv2.resize(face, img_size)
#     face = face.astype('float32') / 255.0
#     return np.expand_dims(face, axis=0)

# def generate_deepfake(img_path, video_path, out_path):
#     src_img = cv2.imread(img_path)
#     src_gray = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
#     src_faces = face_cascade.detectMultiScale(src_gray, 1.3, 5)
#     if len(src_faces) == 0:
#         print("No face found in source image.")
#         return
#     x1, y1, w1, h1 = src_faces[0]
#     src_face = src_img[y1:y1+h1, x1:x1+w1]

#     cap = cv2.VideoCapture(video_path)
#     fourcc = cv2.VideoWriter_fourcc(*'XVID')
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         tgt_faces = face_cascade.detectMultiScale(gray, 1.3, 5)
#         if len(tgt_faces) == 0:
#             out.write(frame)
#             continue

#         x2, y2, w2, h2 = tgt_faces[0]
#         tgt_face = frame[y2:y2+h2, x2:x2+w2]
#         src_input = preprocess_face(src_face)
#         tgt_input = preprocess_face(tgt_face)
#         swapped_face = face_swap_model.predict([src_input, tgt_input])[0]
#         swapped_face_cv = (swapped_face * 255).astype('uint8')
#         swapped_face_cv = cv2.resize(swapped_face_cv, (w2, h2))

#         mask = 255 * np.ones(swapped_face_cv.shape, swapped_face_cv.dtype)
#         center = (x2 + w2//2, y2 + h2//2)
#         try:
#             blended = cv2.seamlessClone(swapped_face_cv, frame, mask, center, cv2.NORMAL_CLONE)
#             out.write(blended)
#         except:
#             out.write(frame)
#     cap.release()
#     out.release()
