import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import gdown

# Descargar modelo desde Drive
MODEL_PATH = "fruit_cnn.tflite"
FILE_ID = "1Gq4U0ubtJaHrfdhfOpQJ3s-4In8PRMhi"
URL = f"https://drive.google.com/uc?id={FILE_ID}"

if not os.path.exists(MODEL_PATH):
    st.write("Descargando modelo desde Google Drive...")
    gdown.download(URL, MODEL_PATH, quiet=False)

# Cargar modelo 
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Clases o categorias
class_names = [
    "apple", "banana", "avocado", "cherry", "kiwi", 
    "mango", "orange", "pineapple", "strawberries", "watermelon"
]

# Función de predicción
def predict(img: Image.Image):
    img = img.resize((180,180))  # mismo tamaño que se uso para entrenar
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_details[0]['index'])[0]
    return preds

# Interfaz Streamlit 
st.title("Clasificador de Frutas con CNN (TFLite)")
st.write("Sube una imagen")

uploaded_file = st.file_uploader("Sube una imagen de fruta", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Imagen subida", use_container_width=True)

    preds = predict(img)
    score = tf.nn.softmax(preds)

    st.write("Predicción:")
    st.write(f"Fruta: **{class_names[np.argmax(score)]}**")
