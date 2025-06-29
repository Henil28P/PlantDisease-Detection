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
