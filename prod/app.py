import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import sys
import os
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import load_model, preprocess_image, predict_image, class_names, get_fruit_freshness_info, get_fruit_info_from_api

# Configuración de la página
st.set_page_config(
    page_title="FruitScan - Freshness Detector",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS personalizados
st.markdown("""
<style>
    .header {
        color: #2e8b57;
        text-align: center;
        font-size: 2.5em;
        margin-bottom: 0.5em;
    }
    .subheader {
        color: #3cb371;
        text-align: center;
        font-size: 1.2em;
        margin-bottom: 2em;
    }
    .result-box {
        border-radius: 10px;
        padding: 1.5em;
        margin: 1em 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .fresh {
        background-color: #e8f5e9;
        border-left: 5px solid #4caf50;
    }
    .rotten {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
    }
    .info-box {
        background-color: #e3f2fd;
        border-left: 5px solid #2196f3;
        border-radius: 10px;
        padding: 1em;
        margin: 1em 0;
    }
    .nutrition-table {
        width: 100%;
        border-collapse: collapse;
    }
    .nutrition-table th {
        background-color: #2e8b57;
        color: white;
        padding: 8px;
        text-align: left;
    }
    .nutrition-table td {
        padding: 8px;
        border-bottom: 1px solid #ddd;
    }
    .nutrition-table tr:nth-child(even) {
        background-color: #f2f2f2;
    }
</style>
""", unsafe_allow_html=True)

# Header de la aplicación
st.markdown('<div class="header">🍏 FruitScan Detector de Frescura</div>', unsafe_allow_html=True)
st.markdown('<div class="subheader">Sube una imagen para comprobar si tu fruta esta fresca o podrida</div>', unsafe_allow_html=True)

# Barra lateral con información
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/415/415733.png", width=100)
    st.markdown("### About")
    st.info("""
    Esta aplicación con AI detecta la frescura de:
    - Manzanas 🍎
    - Bananas 🍌
    - Naranjas 🍊
    
    Sube una foto nitida para mejores resultados.
    """)
    
    st.markdown("### How it works")
    st.write("""
    1. Sube una imagen de una fruta
    2. Nuestro modelo la analiza
    3. Obten resultados de frecura
    4. Aprende sobre la fruta
    """)
    
    st.markdown("### Model Info")
    st.write("""
    - Architecture: ResNet34
    - Accuracy: 94% on test set
    - Last updated: June 2023
    """)

# Carga de imagen
uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"], 
                                help="Sube una foto nitida de una sola fruta (manzana, banana o naranja)")

if uploaded_file is not None:
    try:
        # Mostrar imagen con estilo
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown("### Your Image")
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, use_column_width=True)
        
        with col2:
            with st.spinner('Analyzing your fruit...'):
                # Simular progreso para mejor UX
                progress_bar = st.progress(0)
                for percent_complete in range(100):
                    time.sleep(0.01)
                    progress_bar.progress(percent_complete + 1)
                
                # Cargar modelo (ajusta la ruta según tu estructura)
                model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prod", "modelo.pth")
                num_classes = len(class_names)  # Número de clases en tu modelo
                model = load_model(model_path, num_classes)
                
                # Preprocesar y predecir
                input_tensor = preprocess_image(image)
                predicted_class_name = predict_image(model, input_tensor)
                
                # Mostrar resultados con estilo
                st.markdown("### Analysis Results")
                
                # Determinar clase CSS según resultado
                result_class = "fresh" if "fresh" in predicted_class_name else "rotten"
                result_emoji = "✅" if "fresh" in predicted_class_name else "❌"
                
                st.markdown(f"""
                <div class="result-box {result_class}">
                    <h3>{result_emoji} {predicted_class_name.upper().replace('FRESH', 'FRESH ').replace('ROTTEN', 'ROTTEN ')}</h3>
                    <p>{get_fruit_freshness_info(predicted_class_name)}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Información adicional de la fruta
                st.markdown("### Informacion de la fruta")
                
                # Limpiar nombre para la API
                clean_name = predicted_class_name.replace('fresh', '').replace('rotten', '').strip().lower()
                if clean_name.endswith('s'):
                    clean_name = clean_name[:-1]
                
                api_info = get_fruit_info_from_api(clean_name)
                
                if api_info:
                    # Mostrar información taxonómica
                    with st.expander("📚 Clasificacion Cientifica"):
                        col_tax1, col_tax2 = st.columns(2)
                        with col_tax1:
                            st.markdown(f"""
                            - **Familia**: {api_info.get('family', 'N/A')}
                            - **Orden**: {api_info.get('order', 'N/A')}
                            """)
                        with col_tax2:
                            st.markdown(f"""
                            - **Genero**: {api_info.get('genus', 'N/A')}
                            - **Especie**: {api_info.get('name', 'N/A')}
                            """)
                    
                    # Mostrar información nutricional en tabla
                    with st.expander("🍽️ Nutritional Facts (per 100g)"):
                        nutritions = api_info.get('nutritions', {})
                        st.markdown("""
                        <table class="nutrition-table">
                            <tr>
                                <th>Nutrient</th>
                                <th>Amount</th>
                            </tr>
                            <tr>
                                <td>Calories</td>
                                <td>{calories} kcal</td>
                            </tr>
                            <tr>
                                <td>Carbohydrates</td>
                                <td>{carbs}g</td>
                            </tr>
                            <tr>
                                <td>Sugar</td>
                                <td>{sugar}g</td>
                            </tr>
                            <tr>
                                <td>Protein</td>
                                <td>{protein}g</td>
                            </tr>
                            <tr>
                                <td>Fat</td>
                                <td>{fat}g</td>
                            </tr>
                        </table>
                        """.format(
                            calories=nutritions.get('calories', 'N/A'),
                            carbs=nutritions.get('carbohydrates', 'N/A'),
                            sugar=nutritions.get('sugar', 'N/A'),
                            protein=nutritions.get('protein', 'N/A'),
                            fat=nutritions.get('fat', 'N/A')
                        ), unsafe_allow_html=True)
                    
                    # Consejos según frescura
                    with st.expander("💡 Tips & Recomendaciones"):
                        if "fresh" in predicted_class_name:
                            st.success("**Tips Almacenamiento:**")
                            st.write("- Conservar en un lugar seco y fresco")
                            st.write("- Consumir dentro de un plazao de 3-5 dias para una mejor calidad")
                            st.write("- Refrigerar para extender la frescura")
                        else:
                            st.warning("**Recomendaciones de seguridad:**")
                            st.write("- NO consumir fruta podrida")
                            st.write("- Compostar si es posible")
                            st.write("- Revise frutas cercanas por posible contaminacion")
                else:
                    st.warning("Could not retrieve additional fruit information from the API.")
    
    except FileNotFoundError:
        st.error("Model file not found. Please contact support.")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.info("Please try another image or contact support if the problem persists.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9em;">
    <p>FruitScan Freshness Detector v1.0 | Powered by PyTorch and Streamlit</p>
    <p>For educational purposes only | Not medical advice</p>
</div>
""", unsafe_allow_html=True)
