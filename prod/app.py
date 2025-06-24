import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import sys
import os

# Add the directory containing utils.py to the system path
# Assuming utils.py is in the same directory as app.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import load_model, preprocess_image, predict_image, class_names, get_fruit_freshness_info, get_fruit_info_from_api

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
        model_path = "modelo_resnet34.pth" # Use the correct model filename
        num_classes = len(class_names) # Get the number of classes from utils
        model = load_model(model_path, num_classes)

        # Preprocess the image
        input_tensor = preprocess_image(image)

        # Make a prediction
        predicted_class_name = predict_image(model, input_tensor)

        st.success(f"Prediction: **{predicted_class_name}**")

        # Get freshness information
        freshness_info = get_fruit_freshness_info(predicted_class_name)
        st.markdown(f"## Freshness Information")
        st.write(freshness_info)

        # Get general fruit information from API
        # Clean the predicted name for the API call
        clean_fruit_name_for_api = predicted_class_name.replace('fresh', '').replace('rotten', '')
        if clean_fruit_name_for_api.endswith('s'):
          clean_fruit_name_for_api = clean_fruit_name_for_api[:-1]

        api_info = get_fruit_info_from_api(clean_fruit_name_for_api)

        if api_info:
            st.markdown(f"## General Fruit Information ({api_info.get('name', 'N/A')})")
            st.write(f"Family: {api_info.get('family', 'N/A')}")
            st.write(f"Order: {api_info.get('order', 'N/A')}")
            st.write(f"Genus: {api_info.get('genus', 'N/A')}")
            st.write("Nutritional Information:")
            nutritions = api_info.get('nutritions', {})
            st.write(f"- Calories: {nutritions.get('calories', 'N/A')}")
            st.write(f"- Fat: {nutritions.get('fat', 'N/A')}")
            st.write(f"- Sugar: {nutritions.get('sugar', 'N/A')}")
            st.write(f"- Carbohydrates: {nutritions.get('carbohydrates', 'N/A')}")
            st.write(f"- Protein: {nutritions.get('protein', 'N/A')}")
        else:
            st.warning("Could not retrieve general fruit information from the API.")


    except FileNotFoundError:
        st.error(f"Error: Model file not found at {model_path}. Please ensure the model file is in the correct directory.")
    except Exception as e:
        st.error(f"An error occurred during classification: {e}")
