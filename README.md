# CNN as a Respiratory Diseases Classifier

## Trained on the Respiratory Sound Database using TensorFlow

This notebook establishes an end-to-end machine learning pipeline for classifying respiratory diseases from audio data. By transforming audio signals into visual features, we leverage computer vision techniques to detect abnormalities in lung sounds.

---

## 🛠️ Pipeline Overview

1. **Data Acquisition:** Downloads the 'Respiratory Sound Database' via the Kaggle API and systematically organizes the `.wav` audio files and corresponding diagnosis labels.
2. **Feature Extraction:** Converts raw audio signals into **Mel-Frequency Cepstral Coefficients (MFCCs)** using `librosa`. This transforms sound waves into a 2D "image-like" format ($40 \text{ frequency bands} \times 862 \text{ time frames}$).
3. **Preprocessing & Cleaning:** * Removes extremely rare classes (Asthma and LRTI) to focus on the 6 most common categories (e.g., COPD, Healthy, Pneumonia).
* One-hot encodes the categorical labels.
* Splits the data into stratified training and testing sets.


4. **Modeling:** Defines and trains a custom 2D Convolutional Neural Network (CNN) over 250 epochs.
5. **Evaluation:** Visualizes model performance using ROC curves, Confusion Matrices, and classification reports, achieving high accuracy (particularly for the dominant COPD class).

---

## 🧠 Why This CNN Architecture?

The network architecture was specifically selected to optimize pattern recognition within audio spectrograms while managing a relatively small and imbalanced dataset.

* **2D Convolutions for MFCCs:** Since MFCCs map frequency over time, a CNN is ideal for detecting "spatial" patterns within the spectrogram—such as the specific visual texture of a wheeze versus a normal breath.
* **Pyramidal Filter Depth ($16 \rightarrow 128$):** The network begins with a small number of filters to capture simple audio edges and scales up to 128 filters in deeper layers to capture complex, high-level diagnostic markers.
* **Dimensionality Reduction:** * **MaxPooling** makes the model invariant to *when* a specific sound occurs in the audio clip.
* **GlobalAveragePooling** at the end drastically reduces the total parameter count, preventing the model from simply memorizing the training data.


* **Overfitting Protection:** Incorporates **Dropout (20%)** to ensure the model generalizes well to new, unseen patients. This is critical given that one class (COPD) significantly outweighs the others.
* **Probabilistic Output:** A final **Softmax layer** converts the network's internal signals into clear probability percentages for each of the 6 possible diagnoses.

---

## 📝 Database Context & Content

Respiratory sounds are critical indicators of respiratory health and disorders. The sounds emitted during respiration are directly related to air movement, structural changes within lung tissue, and the presence of secretions. For instance, a **wheezing** sound typically signals an obstructive airway disease like Asthma or Chronic Obstructive Pulmonary Disease (COPD).

By recording these sounds via digital stethoscopes, we can use machine learning to automatically screen for and diagnose conditions like pneumonia, bronchiolitis, and COPD.

### Dataset Composition

Created by research teams in Portugal and Greece, the dataset consists of **920 annotated recordings** (ranging from 10s to 90s) collected from **126 patients**.

* **Total Duration:** 5.5 hours of audio
* **Total Respiratory Cycles:** 6,898 cycles
* 1,864 contain **crackles**
* 886 contain **wheezes**
* 506 contain **both** crackles and wheezes


* **Environment:** Includes both pristine laboratory recordings and noisy, real-life ambient conditions.
* **Demographics:** Spans all age groups (children, adults, and the elderly).

### File Structure & Metadata

The Kaggle dataset includes the following assets:

```yaml
├── 920 .wav sound files
├── 920 annotation .txt files
├── patient_diagnosis.txt         # Diagnosis for each patient
├── filename_format.txt           # File naming convention guide
├── filename_differences.txt      # List of 91 specific filename variations
└── demographic_info.txt          # Patient demographic data

```

#### Demographic Schema

The `demographic_info.txt` file contains 6 distinct features:

1. Patient number
2. Age
3. Sex
4. Adult BMI ($\text{kg/m}^2$)
5. Child Weight ($\text{kg}$)
6. Child Height ($\text{cm}$)

#### Audio Filename Convention

To maintain clinical tracking, each audio file name is programmatically divided into **5 distinct elements**, separated by underscores (`_`), which map the recording back to the patient, session, and equipment used.

---

## 🔗 References & Links

* **Database Research Paper:** [Springer Link - Respiratory Sound Analysis](https://link.springer.com/chapter/10.1007/978-981-10-7419-6_6)
* **Official Repository:** [BHI Challenge Server](https://bhichallenge.med.auth.gr/) or [Alternative Node](https://bhichallenge.med.auth.gr/node/51)
