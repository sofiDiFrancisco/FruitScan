### FruitScan – Detección y segmentación de defectos en frutas con Faster R-CNN

## Propuesta
**Problema:**

Detectar automáticamente defectos visibles en frutas como **bananas, manzanas o peras**, enfocándose en características como **manchas, podredumbre y deformaciones**, utilizando un enfoque basado en **detección de objetos y segmentación**. El objetivo es facilitar la inspección de calidad en frutas de forma automática y visualmente interpretable.

**Dataset propuesto:**

* [Fruit Quality Dataset – Kaggle](https://www.kaggle.com/datasets/sriramr/fruits-fresh-and-rotten-for-classification)


**Aplicación web propuesta:**

Desarrollar una **aplicación web interactiva** (por ejemplo, con **Streamlit**) donde el usuario puede:

* Subir una imagen de una fruta.
* Visualizar los **defectos detectados**, marcados con **bounding boxes** y **máscaras de segmentación**.
* Obtener una breve descripción del tipo de defecto y su localización.

**Objetivo final:** facilitar herramientas automáticas para inspección visual de calidad, con posibles aplicaciones en agricultura, logística y ventas minoristas.



## Estructura
El repositorio de GitHub del proyecto presenta la siguiente estructura:
1. data/: Contendrá los datasets usados para entrenar y evaluar la red neuronal.
2. dev/: Esta carpeta se usará para los notebooks y scripts de desarrollo experimental del modelo. Aquí se incluirán los experimentos en Jupyter notebooks y cualquier script que documente el proceso de exploración y pruebas.
3. prod/:
      ○  app.py: Archivo principal para la aplicación web. Este script contendrá el código para crear la interfaz gráfica con Streamlit y usar la aplicación.
      ○ modelo.pth: Archivo del modelo entrenado que contiene los pesos de la red neuronal en formato PyTorch.
      ○ README.md: Documento de descripción del proyecto que incluye instrucciones sobre cómo clonar el repositorio, instalar las dependencias y ejecutar la aplicación.
      ○ requirements.txt: Archivo con las dependencias necesarias para ejecutar el proyecto.
      ○ utils.py: Archivo con funciones auxiliares utilizadas por app.py, tales como preprocesamiento de datos y carga del modelo.




## Uso
1. Clonar el repositorio:
   ```bash
   git clone https://github.com/pabloMarinozi/ejemploGlobalRNP.git
   ```
2. Instalar dependencias:
   ```bash
   pip install -r requirements.txt
   ```
3. Ejecutar app de streamlit:
   ```bash
   streamlit run prod\app.py
   ```
