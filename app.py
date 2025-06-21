from flask import Flask, request, jsonify, render_template, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
import os
import numpy as np
from PIL import Image
from keras.models import load_model

app = Flask(__name__)

# Set the upload folder for storing the uploaded images
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the pre-trained model
model = load_model('cocoa_disease_model.h5')

# Define the class labels
class_labels = ['class1', 'class2', 'class3', 'class4']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/search', methods=['GET', 'POST'])
def search():
    if request.method == 'POST':
        # Check if the POST request has the file part
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'})

        file = request.files['file']

        # If the user does not select a file, browser also submits an empty part without a filename
        if file.filename == '':
            return jsonify({'error': 'No file selected'})

        # Save the uploaded file to the upload folder
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Redirect to the result page with the filename as a parameter
        return redirect(url_for('result', filename=filename))

    return render_template('search.html')

@app.route('/result/<filename>')
def result(filename):
    # Load and preprocess the input image
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    input_image = Image.open(filepath)
    input_image = input_image.resize((256, 256))
    input_image = np.array(input_image) / 255.0
    input_image = np.expand_dims(input_image, axis=0)

    # Perform image classification and obtain the predicted class
    predictions = model.predict(input_image)
    predicted_class = np.argmax(predictions)
    predicted_class_name = class_labels[predicted_class]

    # Pass the filename, predicted class name, and the file path to the result template
    return render_template('result.html', filename=filename, predicted_class=predicted_class_name, filepath=filename)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
