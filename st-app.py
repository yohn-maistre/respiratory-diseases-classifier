import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import streamlit as st
import tensorflow as tf
from keras.models import load_model

import librosa
import librosa.display
import seaborn as sns

import pdb

st.title('Prediksi Penyakit Pernapasan')
st.write('**Model AI dilatih menggunakan data dengan 6 kategori diagnosis:**')
st.markdown('*-Sehat*   \n*-Bronkiektasis*   \n*-Bronkiolitis*   \n*-Penyakit Paru Obstruktif Kronis (PPOK)*   \n*-Pneumonia*   \n*-Infeksi Saluran Pernapasan Atas*')

uploaded_file = st.file_uploader("Pilih fail audio (hanya format .WAV)")

# Define function to predict
def predict_disease(model, features):
    # Define trained label
    

    # Predict
    prediction = model.predict(features)
    c_pred = np.argmax(prediction)
    
    return c_pred

# Process uploaded Audio
if uploaded_file is not None:
    # Load audio file
    audio, sample_rate = librosa.load(uploaded_file, res_type='kaiser_fast', duration=20)
    # Display waveform
    #if st.checkbox('Display Waveform'):
    #st.write('Waveform')
    #fig, ax = plt.subplots()
    #librosa.display.waveshow(audio, sr=sample_rate)
    #st.pyplot(fig)
    
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
    ######mfccs = np.expand_dims(mfccs, axis=(0, -1))
    max_pad_len = 862
    pad_width = max_pad_len - mfccs.shape[1]
    mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
    features = np.array(mfccs)
    pdb.set_trace()


    if st.button('Prediksi kemungkinan penyakit'):
        c_pred = predict_disease(model, mfccs)
        st.title('Prediksi: ')
        st.subheader(f'**{clabels_idn[c_pred]}**')
        st.subheader(f'*{clabels[c_pred]}*')