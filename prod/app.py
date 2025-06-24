import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import sys
import os

# Add the directory containing utils.py to the system path
# Assuming utils.py is in the same directory as app.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import load_model, preprocess_image, predict_image, class_names, get_fruit_info

st.title("Fruit Freshness Classifier")

st.write("Upload an image of a fruit (apple, banana, or orange) to classify its freshness.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Uploaded Image.", use_column_width=True)
        st.write("")
        st.write("Classifying...")

        # Load the model (ensure the path is correct relative to where app.py will run)
        model_path = "modelo.pth" # Use the correct model filename
        num_classes = len(class_names) # Get the number of classes from utils
        model = load_model(model_path, num_classes)

        # Preprocess the image
        input_tensor = preprocess_image(image)

        # Make a prediction
        predicted_class_name = predict_image(model, input_tensor)

        st.success(f"Prediction: **{predicted_class_name}**")

        # Call the new API function to get additional fruit info
        fruit_info = get_fruit_info(predicted_class_name)

        # Display the retrieved fruit information with improved formatting
        st.markdown(f"## Additional Information")
        st.write(fruit_info)


    except FileNotFoundError:
        st.error(f"Error: Model file not found at {model_path}. Please ensure the model file is in the correct directory.")
    except Exception as e:
        st.error(f"An error occurred during classification: {e}")
