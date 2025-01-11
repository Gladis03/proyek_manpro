from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image

app = Flask(__name__)

# Load model
model = tf.keras.models.load_model("model/model.h5")

# Fungsi preprocessing gambar
def preprocess_image(image):
    image = image.resize((224, 224))  # Sesuaikan dengan model Anda
    image = np.array(image) / 255.0  # Normalisasi
    image = np.expand_dims(image, axis=0)  # Tambahkan dimensi batch
    return image

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    try:
        image = Image.open(file)
        processed_image = preprocess_image(image)
        prediction = model.predict(processed_image)
        class_idx = np.argmax(prediction[0])
        class_name = f"Class {class_idx}"  # Ganti sesuai mapping
        return jsonify({"prediction": class_name})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
