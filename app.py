import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the trained model
# model = tf.keras.models.load_model('models/VGG19.h5')
model = tf.keras.models.load_model('models/ResNet50.h5')
# model = tf.keras.models.load_model('models/ConvNextBase.h5')

# Define classes
classes = ['Cashew Anthrocnase', 
           'Cashew Gumosis',
           'Cashew Healthy',
           'Cashew Lead Miner',
           'Cashew Red Rust',
           'Cassava Bacterial Blight',
           'Cassava Brown Spot',
           'Cassava Green Mite',
           'Cassava Healthy',
           'Cassava Mosaic',
           'Maize Fall Armyworm',
           'Maize Grasshoper'
           'Maize Healthy',
           'Maize Leaf Beetle',
           'Maize Leaf Blight',
           'Maize Leaf Spot',
           'Maize Streak Virus',
           'Tomato Healthy',
           'Tomato Late Blight',
           'Tomato Leaf Curl',
           'Tomato Septoria Leaf Spot',]

# Define a function to preprocess the image
def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image)
    image = image / 255.0  # Normalize pixel values
    return image

# Define a function for making predictions
def predict(image):
    processed_image = preprocess_image(image)
    processed_image = np.expand_dims(processed_image, axis=0)
    predictions = model.predict(processed_image)
    predicted_class = classes[np.argmax(predictions)]
    confidence = np.max(predictions) # Get the confidence score
    return predicted_class, confidence

st.title('Crop Disease Classification')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    predicted_class, confidence = predict(image)

    # Display the prediction
    st.write('Prediction:', predicted_class)
    st.write('Confidence:', confidence)
