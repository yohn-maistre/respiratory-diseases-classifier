import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import streamlit as st
import tensorflow as tf
from keras.models import load_model

import librosa
import librosa.display
import seaborn as sns

#import sounddevice as sd
#from scipy.io.wavfile import write

#import pdb

st.title('Prediksi Penyakit Saluran Pernapasan')
st.caption('*Made with ❤️  by Yose Marthin Giyay*')

st.subheader('**Terdapat 6 kategori diagnosis:**')
st.markdown('*- Sehat*   \n*- Bronkiektasis*   \n*- Bronkiolitis*   \n*- Penyakit Paru Obstruktif Kronis (PPOK)*   \n*- Pneumonia*   \n*- Infeksi Saluran Pernapasan Atas*')
st.subheader('Unggah fail audio dan mulai prediksi')
st.caption('*Dalam pengembangan: rekam langsung*. **Untuk sekarang, silakan unggah fail audio .wav berdurasi ~20 detik**.')

# Define function to predict
def predict_disease(model, features):
    # Predict
    prediction = model.predict(features)
    c_pred = np.argmax(prediction)
    
    return prediction, c_pred

uploaded_file = st.file_uploader("Pilih fail audio (hanya format .WAV)")

# Process uploaded Audio
if uploaded_file is not None:
    # Load audio file
    audio, sample_rate = librosa.load(uploaded_file, duration=20)
    
    # Display spectrogram
    st.write('Spectrogram')
    fig, ax = plt.subplots()
    sns.heatmap(librosa.power_to_db(librosa.feature.melspectrogram(y=audio, sr=sample_rate), ref=np.max))
    st.pyplot(fig)

    # Load model
    model = load_model('./model/CNN-MFCC.h5')

    # Labels
    clabels = ['Bronchiectasis', 'Bronchiolitis', 'Chronic Obstructive Pulmonary Disease (COPD)', 'Healthy', 'Pneumonia', 'Upper Respiratory Tract Infection (URTI)']
    clabels_idn = ['Bronkiektasis', 'Bronkiolitis', 'Penyakit Paru Obstruktif Kronis (PPOK)', 'Sehat', 'Radang Paru-Paru', 'Infeksi Saluran Pernapasan Atas']
    
    # Extract MFCC features from audio clip & 
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    # Add dimensions, (20, 862) to (1, 40, 862, 1)
    max_pad_len = 862
    pad_width = max_pad_len - mfccs.shape[1]
    mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
    features = np.expand_dims(np.array(mfccs), axis=(0, -1))

    if st.button('Prediksi kemungkinan penyakit'):
        prediction, c_pred = predict_disease(model, features)
        max_value = np.max(prediction)
        formatted_max = np.format_float_positional(max_value*100, precision=2)
        st.title('Prediksi: ')
        st.subheader(f'**{clabels_idn[c_pred]}**: {formatted_max}%')
        st.subheader(f'*{clabels[c_pred]}*')

st.markdown('**Konteks Model AI dan Database:**')
st.caption('Model yang digunakan merupakan Convolutional Neural Network (CNN) yang dilatih menggunakan **TensorFlow 2.11.0**. Model ini dilatih dengan data dari **Respiratory Sound Database** yang dikemas 2 tim peneliti di Portugal dan Yunani atas nama *International Conference on Biomedical Health Informatics* (ICHBI)')
st.caption('Di sini _library_ **Librosa** digunakan untuk ekstraksi MFCCs dari fail audio. MFCC, atau Mel-Frequency Cepstral Coefficients, merupakan format representasi audio. Dengan proses matematis ini, fitur-fitur penting di frekuensi alami telinga manusia dapat diekstraksi dari fail audio dan dijadikan *input* ke model CNN untuk diprediksi.')
st.caption('Database yang digunakan dapat dijelajahi dan/atau diunduh di sini: https://bhichallenge.med.auth.gr/')
st.caption('Jurnal ilmiah menyangkut database dapat dilihat di sini: https://link.springer.com/chapter/10.1007/978-981-10-7419-6_6')