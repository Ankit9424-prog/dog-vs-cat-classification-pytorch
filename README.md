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
dog-vs-cat-classification-pytorch/
├── notebook/
│   └── dog_vs_cat_classification.ipynb
├── dogvscat.py
├── requirements.txt
└── README.md


## 📊 Dataset
- Dataset: **Microsoft Dogs vs Cats Dataset (Kaggle)**
- Classes: Dog, Cat
- Invalid/corrupted images are automatically skipped
- Images are normalized using ImageNet mean & std

⚠️ The dataset is **not included** in the repository.

## ⚙️ Installation
```bash
pip install -r requirements.txt
```
## 🚀 How to Run
## Option 1: Google Colab (Recommended)
Upload the notebook to Google Colab
Run all cells
Dataset downloads automatically using kagglehub

## Option 2: Local Machine
python dogvscat.py

## 📈 Training Details
Epochs: 5
Batch size: 32
Optimizer: Adam
Loss function: CrossEntropyLoss
Device: CPU / CUDA (auto-detected)

## 📊 Evaluation
Accuracy and loss tracked per epoch
Confusion matrix generated using scikit-learn
Random test image prediction with confidence score

## 🛠 Tools & Libraries
Python=
PyTorch
Torchvision
NumPy
Matplotlib
scikit-learn
Google Colab

## 🔮 Future Improvements
Modularize code (model.py, train.py)
Add validation curves
Use transfer learning (ResNet, MobileNet)
Deploy using Streamlit or FastAPI


