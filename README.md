## Author

Gülseren Merve Şafak  
Bachelor Thesis – 2025  
Supervisor: Dr. Murat Kırtay
# Supervised Anomaly Detection in Mars Rover Telemetry Using Machine Learning and Deep Learning Models

This repository contains the complete implementation and results of my bachelor thesis on supervised anomaly detection in Mars rover telemetry data. The aim of the project is to detect faults in spacecraft telemetry using sliding-window-based models trained on labeled time series from the MSL (Mars Science Laboratory) (Hundman et al., 2018) dataset.

## Overview

Rovers like Curiosity operate autonomously and rely on robust onboard systems. Fault detection is critical for maintaining safe operations. This project uses labeled anomaly intervals from MSL telemetry (Hundman et al., 2018) to train supervised learning models that can classify time windows as anomalous or normal.


- Random Forest (used as a baseline)
- 1D Convolutional Neural Network (CNN)
- CNN-LSTM Hybrid model (for combining spatial and temporal information)

All models work on fixed-size sliding windows (length = 100) extracted from telemetry channels.

## Contents

This repository includes:

- Three Jupyter notebooks (`cnn_model`, `cnn_lstm_model`, and `random_forest_model`) under the `notebooks/` directory
- Labeled anomaly data (`labeled_anomalies.csv`) and `.npy` telemetry files under `data/`
- Saved model predictions and summaries (`.npz` and `.json`) under `outputs/`
- Evaluation visualizations (overall ROC curves, PR curves, confusion matrices and per channel visualizations) under `plots/`
- Saved model weights (`.h5` files) under `models/`
- This README file and a `.gitignore`

## Dataset

This work uses the MSL portion of the anomaly detection dataset introduced by:

Hundman, K., Constantinou, V., Laporte, C., Colwell, I., & Soderstrom, T. (2018).  
**Detecting Spacecraft Anomalies Using LSTMs and Nonparametric Dynamic Thresholding.**  
*Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining*, 387–395.  
https://doi.org/10.1145/3219819.3219845

Only MSL channels were used, with anomaly labels matched against fixed-length windows.
## Environment and Packages

This project was developed and tested on Google Colab with GPU support. The main packages and versions used:

- Python 3.10+
- NumPy 1.23+
- pandas 1.5+
- scikit-learn 1.3+
- TensorFlow 2.13+
- matplotlib 3.7+
  
## Implementation Notes

- Focal loss (Lin et al., 2017) was used in deep learning models to address the extreme class imbalance.
- Class weights were computed in Random Forest and CNN models to reduce bias toward the majority class.
- Model evaluation is done using precision, recall, F1 score, ROC AUC, and PR AUC — both per channel and overall (aggregated test results).
- Early stopping and checkpointing were applied to prevent overfitting.

## External Resources and Support

Several parts of this implementation were supported by posts on Stack Overflow, these references are cited directly in the code.

Final code polishing, structure cleanup, and documentation formatting were supported using OpenAI's ChatGPT.




