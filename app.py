import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import gdown

# --- Descargar modelo desde Google Drive si no existe ---
MODEL_PATH = "fruit_cnn.tflite"
FILE_ID = "1Gq4U0ubtJaHrfdhfOpQJ3s-4In8PRMhi"
URL = f"https://drive.google.com/uc?id={FILE_ID}"

if not os.path.exists(MODEL_PATH):
    st.write("📥 Descargando modelo desde Google Drive...")
    gdown.download(URL, MODEL_PATH, quiet=False)

# --- Cargar modelo TFLite ---
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# --- Clases (ajusta con tus categorías reales) ---
class_names = [
    "apple", "banana", "cherry", "date", "grape", 
    "kiwi", "lemon", "mango", "orange", "watermelon"
]

# --- Función de predicción ---
def predict(img: Image.Image):
    img = img.resize((180,180))  # mismo tamaño que usaste para entrenar
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_details[0]['index'])[0]
    return preds

# --- Interfaz Streamlit ---
st.title("🍉 Clasificador de Frutas con CNN (TFLite)")
st.write("Sube una imagen y el modelo intentará adivinar qué fruta es.")

uploaded_file = st.file_uploader("📷 Sube una imagen de fruta", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Imagen subida", use_container_width=True)

    preds = predict(img)
    score = tf.nn.softmax(preds)

    st.write("### 🔮 Predicción:")
    st.write(f"Fruta: **{class_names[np.argmax(score)]}**")
    st.write(f"Confianza: **{100*np.max(score):.2f}%**")
