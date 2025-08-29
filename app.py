import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# --- Cargar modelo ---
@st.cache_resource
def load_cnn():
    model = load_model("fruit_cnn.h5")
    return model

model = load_cnn()

# --- Definir categorías ---
# OJO: usa la misma lista que tenías en `data_cat` al entrenar
class_names = [
    "apple", "banana", "cherry", "date", "grape", 
    "kiwi", "lemon", "mango", "orange", "watermelon"
]

# --- Interfaz Streamlit ---
st.title("🍉 Clasificador de Frutas con CNN")
st.write("Sube una imagen y el modelo intentará adivinar la fruta.")

# Subir imagen
uploaded_file = st.file_uploader("Sube una imagen de fruta", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Imagen subida", use_container_width=True)

    # Preprocesamiento (mismo tamaño que usaste para entrenar)
    img_resized = img.resize((180,180))  
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Predicción
    preds = model.predict(img_array)
    score = tf.nn.softmax(preds[0])
    
    st.write("### 🔮 Predicción:")
    st.write(f"Fruta: **{class_names[np.argmax(score)]}**")
    st.write(f"Confianza: **{100*np.max(score):.2f}%**")
