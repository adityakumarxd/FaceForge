from flask import Flask, render_template, request, redirect, url_for, send_from_directory, flash
import os
from werkzeug.utils import secure_filename
from deepfake_generation.generator import generate_deepfake
from deepfake_detection.detector import detect_deepfake

UPLOAD_FOLDER = './static/uploads'
ALLOWED_EXTENSIONS_IMAGE = {'png', 'jpg', 'jpeg'}
ALLOWED_EXTENSIONS_VIDEO = {'mp4', 'avi', 'mov'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'supersecretkey'

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename, allowed_set):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_set

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        image = request.files.get('image')
        video = request.files.get('video')
        if not image or not video:
            flash('Please upload BOTH image and video.')
            return redirect(request.url)
        if not allowed_file(image.filename, ALLOWED_EXTENSIONS_IMAGE):
            flash('Image format not supported. Use png/jpg/jpeg.')
            return redirect(request.url)
        if not allowed_file(video.filename, ALLOWED_EXTENSIONS_VIDEO):
            flash('Video format not supported. Use mp4/avi/mov.')
            return redirect(request.url)
        image_filename = secure_filename(image.filename)
        video_filename = secure_filename(video.filename)
        image.save(os.path.join(app.config['UPLOAD_FOLDER'], image_filename))
        video.save(os.path.join(app.config['UPLOAD_FOLDER'], video_filename))
        return redirect(url_for('results', video_filename=video_filename, image_filename=image_filename))
    return render_template('upload.html')

@app.route('/results')
def results():
    image_filename = request.args.get('image_filename')
    video_filename = request.args.get('video_filename')
    generated_filename = None
    prediction = None
    confidence = None

    # Run generator
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_filename)
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_filename)
    generated_filename = video_filename.replace('.', '_deepfaked.')
    generated_path = os.path.join(app.config['UPLOAD_FOLDER'], generated_filename)
    if not os.path.exists(generated_path):
        generate_deepfake(image_path, video_path, generated_path)

    # Run detector
    prediction, confidence = detect_deepfake(generated_path)

    return render_template('results.html',
                           video_filename=video_filename,
                           image_filename=image_filename,
                           generated_filename=generated_filename,
                           prediction=prediction,
                           confidence=confidence)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == "__main__":
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)
