# app.py - Flask Application for Eye Disease Detection

from flask import Flask, request, render_template, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import preprocess_input
import numpy as np
import os

# Load the trained model (ensure evgg.h5 exists from model_training.py)
model = load_model("evgg.h5")

app = Flask(__name__)

# Create the 'uploads' directory if it doesn't exist
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict', methods=["POST"])
def res():
    f = request.files['image']
    filepath = os.path.join(UPLOAD_FOLDER, f.filename)
    f.save(filepath)
    
    # Load and preprocess the image
    img = image.load_img(filepath, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    img_data = preprocess_input(x)
    
    # Make prediction using the model
    prediction = np.argmax(model.predict(img_data), axis=1)
    classes = ['Cataract', 'Diabetic Retinopathy', 'Glaucoma', 'Normal']
    result = str(classes[prediction[0]])
    
    # Return JSON response for AJAX requests
    return jsonify({"prediction": result})

if __name__ == "__main__":
    app.run(debug=True)
