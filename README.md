# 🧠 Brain Tumor AI Medical Intelligence Platform

> An end-to-end AI-powered medical imaging platform for brain tumor detection from MRI scans using Deep Learning, Explainable AI (Grad-CAM), FastAPI, and Streamlit. The platform provides accurate predictions, interpretable visualizations, AI-assisted medical reports, and prediction history management.

---

## Table of Contents

- Overview
- Features
- System Architecture
- Project Structure
- Tech Stack
- Dataset
- Model Development
- Performance
- Explainable AI (Grad-CAM)
- REST API
- Streamlit Dashboard
- Installation
- Usage
- Future Enhancements
- License
- Author

---

# Overview

Brain tumor diagnosis from MRI scans is a critical and time-sensitive task. Manual interpretation requires significant expertise and can be time-consuming. This project demonstrates how Deep Learning and Explainable AI can assist clinicians by automating tumor classification while providing visual explanations for model predictions.

The platform combines a trained EfficientNet-B0 model with Grad-CAM visualizations, AI-generated medical reports, and an intuitive web interface to deliver a complete end-to-end medical AI workflow.

---

# Features

- Brain MRI Tumor Classification
- EfficientNet-B0 Deep Learning Model
- Explainable AI using Grad-CAM
- AI-Assisted Medical Report Generation
- PDF Report Export
- FastAPI REST API
- Interactive Streamlit Dashboard
- SQLite Prediction History
- Docker Support
- Modular & Production-Ready Project Structure

---

# System Architecture


                    MRI Image
                        │
                        ▼
              Image Preprocessing
                        │
                        ▼
              EfficientNet-B0 Model
                        │
        ┌───────────────┴───────────────┐
        ▼                               ▼
 Prediction                    Grad-CAM Heatmap
        │                               │
        └───────────────┬───────────────┘
                        ▼
            AI Medical Report Generator
                        │
                        ▼
              SQLite Prediction History
                        │
                        ▼
              Streamlit Web Application


# Project Structure


brain-tumor-ai-platform/
│
├── app/                  # Streamlit application
├── data/                 # Dataset
├── docs/                 # Documentation & images
├── models/               # Trained models
├── notebooks/            # Jupyter notebooks
├── outputs/              # Reports & Grad-CAM outputs
├── src/
│   ├── api/
│   ├── data/
│   ├── db/
│   ├── llm/
│   ├── models/
│   ├── utils/
│   └── xai/
│
├── tests/
├── Dockerfile
├── requirements.txt
└── README.md


# Tech Stack

| Category | Technologies |
|-----------|--------------|
| Programming | Python |
| Deep Learning | PyTorch, EfficientNet-B0 |
| Explainable AI | Grad-CAM |
| Backend | FastAPI |
| Frontend | Streamlit |
| Database | SQLite |
| Reporting | ReportLab |
| Visualization | Matplotlib |
| Image Processing | Pillow, OpenCV |
| Deployment | Docker |


# Dataset

**Dataset:** Brain MRI Images

The dataset consists of MRI brain scan images categorized into multiple tumor classes and normal cases.

### Data Pipeline

- Data collection
- Image preprocessing
- Data augmentation
- Train-validation split
- Model training
- Evaluation

## Data Source: 
https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset



# Model Development

### Workflow

1. Data preprocessing
2. Image augmentation
3. EfficientNet-B0 training
4. Model evaluation
5. Prediction
6. Explainability using Grad-CAM
7. AI medical report generation

### Training Highlights

- Transfer Learning using EfficientNet-B0
- Cross-Entropy Loss
- Adam Optimizer
- Learning Rate Scheduling
- Validation Monitoring


# Performance

Brain Tumor AI Model Evaluation Summary
==================================================

Model: EfficientNet-B0
Dataset: Brain Tumor MRI Dataset
Number of Classes: 4
Classes: ['glioma', 'meningioma', 'notumor', 'pituitary']
Best Validation Accuracy (%): 95.62
Macro Precision: 0.9597
Macro Recall: 0.9563
Macro F1-Score: 0.9552
Weighted Precision: 0.9597
Weighted Recall: 0.9563
Weighted F1-Score: 0.9552

### Evaluation Outputs

- Classification Report
- Confusion Matrix
- ROC Curve
- Accuracy/Loss Curves

---

# Explainable AI (Grad-CAM)

To improve model transparency, the platform integrates **Gradient-weighted Class Activation Mapping (Grad-CAM)**.

Grad-CAM highlights the regions of the MRI scan that contributed most to the model's prediction, helping users better understand the decision-making process.


# 🌐 REST API

| Method | Endpoint | Description |
|----------|----------|-------------|
| GET | `/health` | API health check |
| POST | `/predict` | Predict brain tumor |
| POST | `/gradcam` | Generate Grad-CAM visualization |
| POST | `/report` | Generate AI medical report |
| POST | `/report/pdf` | Download report as PDF |
| GET | `/predictions` | Retrieve prediction history |
| DELETE | `/predictions/{id}` | Delete prediction record |

---

# Streamlit Dashboard

The web application provides a user-friendly interface for interacting with the AI model.

### Dashboard Features

- MRI Image Upload
- Prediction Results
- Confidence Scores
- Grad-CAM Visualization
- AI Medical Report
- PDF Download
- Prediction History


# Installation

Clone the repository

```bash
git clone https://github.com/ManojDursoju/brain-tumor-ai-platform.git

cd brain-tumor-ai-platform
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the FastAPI server

```bash
uvicorn src.api.main:app --reload
```

Run the Streamlit application

```bash
streamlit run app/streamlit_app.py
```

---

# Usage

1. Launch the FastAPI backend.
2. Start the Streamlit application.
3. Upload an MRI brain scan.
4. View the predicted class and confidence score.
5. Generate a Grad-CAM explanation.
6. Create an AI-assisted medical report.
7. Export the report as a PDF.
8. Review previous predictions from the history page.

---

# Future Enhancements

- Multi-class tumor classification
- MRI segmentation models
- DICOM image support
- Cloud deployment (AWS/Azure/GCP)
- User authentication
- Doctor/Admin dashboard
- Model monitoring and logging
- CI/CD pipeline integration

---

# License

This project is licensed under the MIT License.

---

# Author

**Manoj Dursoju**

- GitHub: https://github.com/ManojDursoju
- LinkedIn: https://www.linkedin.com/in/manojdursoju

---

### If you found this project useful, consider giving it a star on GitHub.