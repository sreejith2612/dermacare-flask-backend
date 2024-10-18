from flask import Flask, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import io

app = Flask(__name__)

# Load your pre-trained h5 model
model = load_model('my_model.h5')

# Define class labels (replace with actual class labels)
class_labels = ['Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5', 'Class 6', 'Class 7']

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']

    try:
        # Open the image and resize to 75x100
        image = Image.open(io.BytesIO(file.read()))
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
