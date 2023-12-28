import streamlit as st
from keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img
import numpy as np

st.title("Banana Detection App")
model = load_model('cnn_model.h5')

# Definisikan label kelas yang sesuai dengan model Anda
class_labels = ["matang", "mentah", "terlalu matang"]  # Ganti dengan label kelas yang sesuai

# Fungsi prediksi
def predict_banana(image):
    target_size = (150, 150)
    img = load_img(image, target_size=target_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    prediction = model.predict(img_array)
    predicted_class = class_labels[np.argmax(prediction)]
    confidence = float(np.max(prediction))
    return predicted_class, confidence

# Mengunggah gambar dari pengguna
uploaded_file = st.file_uploader("Masukan Gambar Banana...", type="jpg")

# Menampilkan prediksi jika gambar diunggah
if uploaded_file is not None:
    predicted_class, confidence = predict_banana(uploaded_file)

    # Menampilkan gambar dan hasil prediksi
    st.image(load_img(uploaded_file), caption='Uploaded Image', use_column_width=True)
    st.write(f'Prediction: {predicted_class}')
    progress_bar = st.progress(confidence)
    st.text(f'Confidence: {confidence:.2%}')
