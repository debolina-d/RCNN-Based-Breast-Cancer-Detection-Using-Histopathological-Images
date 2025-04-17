# RCNN-Based Breast Cancer Detection Using Histopathological Images 🧠🔬

This project implements a **Region-based Convolutional Neural Network (RCNN)** to detect breast cancer from histopathological images using **ResNet50** as a feature extractor. It uses the **BreakHis dataset** as the image source.

---

## 📌 Features

- ✅ Transfer learning with pre-trained ResNet50
- ✅ Region Proposal Network (RPN)
- ✅ Dual outputs: 
  - Binary classification (Benign/Malignant)
  - Bounding box regression
- ✅ Data augmentation using Keras' `ImageDataGenerator`
- ✅ EarlyStopping and Learning Rate Scheduling
- ✅ Deployment-ready Flask app (`app.py`)

---

## 📊 Dataset: BreakHis

- **Name:** BreakHis (Breast Cancer Histopathological Database)
- **Source:** [BreakHis Official Website](https://www.kaggle.com/datasets/ambarish/breakhis)
- **Classes:** Benign and Malignant tumors
- **Image Magnifications:** 40x, 100x, 200x, 400x


---

## ⚙️ Installation & Setup

1. Clone the repository:
```bash
git clone https://github.com/debolina-d/RCNN-Based-Cancer-Detection-Histopathological-Images.git
cd RCNN-Based-Cancer-Detection-Histopathological-Images



