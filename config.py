
class Config:
    UPLOAD_FOLDER = 'static/uploads'
    ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # Limit upload size to 16 MB

    # GENERATION_MODEL_PATH = 'deepfake_generation/models/generator_model.h5'
    
    # DETECTION_MODEL_PATH = 'deepfake_detection/models/detector_model.h5'