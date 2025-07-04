import os
import json
from PIL import Image

import numpy as np
import tensorflow as tf
import streamlit as st

current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = f"{current_dir}/Trained Model/plant_disease_prediction_model.h5"
# Load the pre-trained model
model = tf.keras.models.load_model(model_path)

# loading the class names
class_indices = json.load(open(f"{current_dir}/class_indices.json"))

# Function to load and preprocess the Image using Pillow
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path) # load the image
    img = img.resize(target_size) # resize the image
    img_array = np.array(img) # convert the image to a numpy array
    img_array = np.expand_dims(img_array, axis=0) # add batch dimension
    img_array = img_array.astype('float32') / 255. # scale the image values to [0,1]
    return img_array

# TensorFlow model prediction function
def predict_image_class(model, image_path, class_indices):
    preprocessed_img = load_and_preprocess_image(image_path)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0] # returns the index of the highest probability value
    predicted_class_name = class_indices[predicted_class_index] # gets the name of that class with highest probability
    return predicted_class_name
