---
title: Respiratory Sound Classification
emoji: 🫁
colorFrom: blue
colorTo: red
sdk: streamlit
python_version: 3.10
app_file: app.py
pinned: false
---

# Trained on the Respiratory Sound Database using TensorFlow

Database paper can be found here: 
https://link.springer.com/chapter/10.1007/978-981-10-7419-6_6

Database can be downloaded/explored [here](https://bhichallenge.med.auth.gr/) or [here](https://bhichallenge.med.auth.gr/node/51)

# Context
Respiratory sounds are important indicators of respiratory health and respiratory disorders. The sound emitted when a person breathes is directly related to air movement, changes within lung tissue and the position of secretions within the lung. A wheezing sound, for example, is a common sign that a patient has an obstructive airway disease like asthma or chronic obstructive pulmonary disease (COPD).

These sounds can be recorded using digital stethoscopes and other recording techniques. This digital data opens up the possibility of using machine learning to automatically diagnose respiratory disorders like asthma, pneumonia and bronchiolitis, to name a few. Content

The Respiratory Sound Database was created by two research teams in Portugal and Greece. It includes 920 annotated recordings of varying length - 10s to 90s. These recordings were taken from 126 patients. There are a total of 5.5 hours of recordings containing 6898 respiratory cycles - 1864 contain crackles, 886 contain wheezes and 506 contain both crackles and wheezes. The data includes both clean respiratory sounds as well as noisy recordings that simulate real life conditions. The patients span all age groups - children, adults and the elderly.

This Kaggle dataset includes:
```javascript
920 .wav sound files
920 annotation .txt files
A text file listing the diagnosis for each patient
A text file explaining the file naming format
A text file listing 91 names (filename_differences.txt)
A text file containing demographic information for each patient
```

The demographic info file has 6 columns:

Patient number
Age
Sex
Adult BMI (kg/m2)
Child Weight (kg)
Child Height (cm)
Each audio file name is divided into 5 elements, separated with underscores (_).
