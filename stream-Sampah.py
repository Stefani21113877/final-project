import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array
import tensorflow as tf
import streamlit as st
import os

# Muat model terbaik yang sudah dilatih
model = tf.keras.models.load_model('C:\CNN-CLASIFICATION SAMPAH\yak_model.h5')

# Kelas yang digunakan untuk prediksi
# Anda perlu memastikan classes di-load dari tempat yang benar
# Misalnya, bisa diambil dari pickle file atau hardcode seperti berikut ini:
classes = ['kertas', 'organik', 'plastik']  # Ganti dengan nama kelas sebenarnya
print(classes)

# Fungsi untuk memuat dan memproses gambar
def load_and_process_image(image_path, img_size=(224, 224)):
    image = Image.open(image_path).convert("RGB")
    image = image.resize(img_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0  # Rescale
    return image

# Fungsi untuk memprediksi gambar
def predict_image(image_path):
    image = load_and_process_image(image_path)
    prediction = model.predict(image)
    predicted_class = np.argmax(prediction, axis=-1)
    predicted_class_name = classes[predicted_class[0]]
    return predicted_class_name, prediction[0]

# Streamlit App
st.markdown(
    """
    <style>
    .main {
        background-color: #f5f5f5;
        padding: 20px;
        border-radius: 10px;
    }
    .title {
        color: #2c3e50;
        text-align: center;
    }
    .uploaded-image {
        text-align: center;
        margin-top: 20px;
    }
    .result {
        text-align: center;
        margin-top: 20px;
        font-size: 24px;
        color: #27ae60;
    }
    .probabilities {
        text-align: center;
        margin-top: 20px;
    }
    .probability-bar {
        background-color: #ecf0f1;
        border-radius: 10px;
        margin: 5px 0;
    }
    .bar {
        height: 24px;
        border-radius: 10px;
        text-align: center;
        color: white;
        line-height: 24px;
    }
    </style>
    """, unsafe_allow_html=True
)

st.markdown('<h1 class="title">Image Classification Sampah </h1>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True, output_format='JPEG')
    st.write("")
    st.write("Classifying...")
    
    # Simpan gambar yang diupload ke file sementara
    temp_file_path = "temp_uploaded_image.jpg"
    image.save(temp_file_path)
    
    # Prediksi gambar
    predicted_class_name, prediction_probabilities = predict_image(temp_file_path)
    
    # Hapus file sementara
    os.remove(temp_file_path)
    
    st.markdown(f'<div class="result">Prediksi: {predicted_class_name}</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="probabilities">Probabilitas prediksi untuk setiap kelas:</div>', unsafe_allow_html=True)
    for i, class_name in enumerate(classes):
        probability = prediction_probabilities[i] * 100
        bar_color = "#3498db" if probability > 50 else "#e74c3c"
        st.markdown(
            f'<div class="probability-bar"><div class="bar" style="width: {probability}%; background-color: {bar_color}">{class_name}: {probability:.2f}%</div></div>',
            unsafe_allow_html=True
        )
