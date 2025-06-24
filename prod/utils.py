import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import sys
import os

# Configuración de rutas
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from utils import load_model, preprocess_image, predict_image, class_names, get_fruit_freshness_info, get_fruit_info_from_api

# Configuración de la aplicación
st.set_page_config(page_title="Fruit Freshness Classifier", layout="wide")
st.title("🍓 Fruit Freshness Classifier")
st.write("Upload an image of a fruit (apple, banana, or orange) to classify its freshness.")

# Widget para subir imagen
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Mostrar imagen subida
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Uploaded Image.", use_column_width=True)
        
        # Cargar modelo
        model_path = os.path.join(current_dir, "prod", "modelo.pth")
        if not os.path.exists(model_path):
            st.error(f"Model file not found at {model_path}")
            st.stop()
            
        model = load_model(model_path, len(class_names))
        
        # Preprocesar y predecir
        input_tensor = preprocess_image(image)
        predicted_class_name = predict_image(model, input_tensor)
        
        # Mostrar resultados
        st.success(f"**Prediction:** {predicted_class_name}")
        
        # Información de frescura
        freshness_info = get_fruit_freshness_info(predicted_class_name)
        st.markdown("### 🍃 Freshness Information")
        st.info(freshness_info)
        
        # Información general de la fruta
        clean_name = predicted_class_name.replace('fresh', '').replace('rotten', '').strip().lower()
        if clean_name.endswith('s'):
            clean_name = clean_name[:-1]
            
        api_info = get_fruit_info_from_api(clean_name)
        
        if api_info:
            st.markdown(f"### ℹ️ General Information about {api_info.get('name', 'this fruit')}")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Classification:**")
                st.write(f"- Family: {api_info.get('family', 'N/A')}")
                st.write(f"- Order: {api_info.get('order', 'N/A')}")
                st.write(f"- Genus: {api_info.get('genus', 'N/A')}")
            
            with col2:
                st.write("**Nutritional Information (per 100g):**")
                nutritions = api_info.get('nutritions', {})
                st.write(f"- Calories: {nutritions.get('calories', 'N/A')} kcal")
                st.write(f"- Fat: {nutritions.get('fat', 'N/A')}g")
                st.write(f"- Sugar: {nutritions.get('sugar', 'N/A')}g")
                st.write(f"- Carbohydrates: {nutritions.get('carbohydrates', 'N/A')}g")
                st.write(f"- Protein: {nutritions.get('protein', 'N/A')}g")
        else:
            st.warning("Could not retrieve additional fruit information from the API.")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
