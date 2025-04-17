from flask import Flask, render_template, request, redirect, url_for
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

MODEL_PATH = r"C:\Users\hp\Documents\Notes\MCA\2nd Sem\Machine Learning\Project_ML\rcnn_model_3rd.h5"

try:
    model = load_model(MODEL_PATH, custom_objects={'mse': MeanSquaredError()})
    print(" Model loaded successfully!")
except Exception as e:
    print(f" Error loading model: {e}")
    model = None

IMG_WIDTH, IMG_HEIGHT = 224, 224

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        img = cv2.imread(filepath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
        img = np.expand_dims(img, axis=0) / 255.0

        if model:
            prediction = model.predict(img)[0][0]
            label = "Malignant" if prediction >= 0.5 else "Benign"
            return render_template('result.html', label=label, image=filepath)
        else:
            return "Error: Model not loaded!", 500

if __name__ == '__main__':
    app.run(debug=True)
