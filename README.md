# Ear Biometrics Authentication System

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=for-the-badge&logo=tensorflow)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?style=for-the-badge&logo=opencv)
![PyQt5](https://img.shields.io/badge/PyQt5-Desktop%20App-lightgrey?style=for-the-badge&logo=qt)

An advanced authentication system utilizing deep neural networks to verify user identity based on their ear biometrics. 

Note: This project was developed as my engineering thesis and was awarded the highest possible grade (5.5).

## Project Overview

The core objective of this system is to securely add users to a database and authenticate their access using ear biometry. By capturing a live feed, detecting the ear, and extracting a unique feature embedding, the system acts as a reliable biometric lock.

### Key Components

- Application: A user-friendly graphical interface built with PyQt5 for real-time camera capture, user registration, and login.
- Ear Detection Model: Utilizes a fine-tuned YOLO model for accurate, real-time ear detection and cropping from the camera feed.
- Siamese Neural Network: A ResNet50-based embedding model trained with Triplet Loss to generate highly discriminative 1D feature vectors for each ear.
- Data Augmentation: A robust script simulating various environmental conditions (rotation, noise, brightness, random erasing) to enrich the training dataset and improve model generalization.

## How It Works

1. Detection: The system continuously scans the camera feed to detect an ear using YOLO.
2. Extraction: Once an ear is detected and its quality is approved by the user, the image is cropped, resized, and normalized.
3. Embedding: The cropped ear passes through the Siamese network to generate a unique embedding vector.
4. Verification: 
   - Registration: The embedding is securely encrypted and stored in the PostgreSQL database.
   - Authentication: The live embedding is compared against the stored one using Euclidean distance. Access is granted if the distance is below the acceptable threshold (EER).

## Repository Structure

- Application/ : Source code for the PyQt5 desktop application and database integration.
- Notebooks_training_augmentation/ : Jupyter notebooks containing model architecture, training loops, evaluation metrics (t-SNE, ROC-AUC), and the image augmentation script.

## Future Enhancements

While this system serves as a comprehensive proof-of-concept for ear biometrics, future iterations could include liveliness detection and integration with other multi-modal biometric systems.
