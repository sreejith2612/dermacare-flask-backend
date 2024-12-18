from flask import Flask, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import io
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={
    r"/predict": {
        "origins": [
            "https://dermacare26.vercel.app",
            r"http://localhost:\d+"  # Allow any localhost port
        ]
    }
})

# Load your pre-trained h5 model
model = load_model('my_model.h5')

# Updated class labels to match frontend
class_labels = [
    'Melanocytic nevi',
    'Melanoma',
    'Benign keratosis-like lesions',
    'Basal cell carcinoma',
    'Actinic keratoses',
    'Vascular lesions',
    'Dermatofibroma'
]

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']

    try:
        # Open the image
        image = Image.open(io.BytesIO(file.read()))

        # Convert PNG to JPG if necessary
        if image.format == 'PNG':
            image = image.convert('RGB')

        # Resize to 100x75
        image = image.resize((100, 75))

        # Preprocess the image to match the model input requirements
        image_array = np.asarray(image) / 255.0  # Normalize the image
        image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

        # Perform prediction
        predictions = model.predict(image_array)
        predicted_class_index = np.argmax(predictions)
        confidence = predictions[0][predicted_class_index]

        # Prepare response
        result = {
            'predicted_class': class_labels[predicted_class_index],
            'confidence': float(confidence)
        }

        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
