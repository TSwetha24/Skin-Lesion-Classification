from flask import Flask, request, render_template
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import os
from PIL import Image

app = Flask(__name__)

# Load the trained model
MODEL_PATH = "skin_lesion_classifier.keras"
model = load_model(MODEL_PATH)

# Define class labels
CLASS_LABELS = {0.: "🟢 Benign", 1: "🔴 Melanoma"}

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def home():
    return render_template('index.html', result=None, image_path=None)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded", 400

    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400

    # Save the uploaded image
    img_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(img_path)

    # Preprocess the image
    img = Image.open(img_path).convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)[0][0]
    result = CLASS_LABELS[1] if prediction > 0.5 else CLASS_LABELS[0]
    confidence = round(prediction if prediction > 0.5 else 1 - prediction, 4)

    return render_template('index.html', result=f"{result} (Confidence: {confidence})", image_path=img_path)

if __name__ == '__main__':
    app.run(debug=True)