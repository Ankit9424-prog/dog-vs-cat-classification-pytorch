# Dog vs Cat Image Classification using PyTorch CNN

This project implements a **Convolutional Neural Network (CNN) from scratch using PyTorch** to classify images as **Dog** or **Cat**.  
The entire pipeline was developed and trained using **Google Colab**.

## 📌 Project Highlights
- Custom CNN architecture (no transfer learning)
- Image preprocessing and normalization
- Train–test split (80/20)
- Training & evaluation loops
- Prediction on unseen images
- Confusion matrix visualization

## 🧠 Model Architecture
The CNN consists of:
- Two convolutional blocks:
  - Conv2D → ReLU → Conv2D → ReLU → MaxPooling
- Fully connected classifier layer
- CrossEntropyLoss for training
- Adam optimizer

Input images are resized to **224×224**.

## 🗂 Project Structure
