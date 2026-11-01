# Bidepzai – Audio Emotion Analysis

## Overview

Bidepzai is a machine learning-based audio analysis website designed to analyze audio input and predict emotional characteristics using a trained classification model.

The project combines a web-based interface with machine learning models to provide an interactive audio analysis experience.

## Features

- Audio analysis through a web interface
- Machine learning-based emotion classification
- Text-based emotion analysis using a trained Logistic Regression model
- TF-IDF feature extraction
- Interactive results displayed through the website

## Machine Learning

The project uses the following trained machine learning components:

- Logistic Regression classifier
- TF-IDF vectorizer
- Label encoder

These components are stored as serialized `.pkl` files and loaded by the application for prediction.

## Technologies

- Python
- Machine Learning
- Scikit-learn
- TF-IDF
- Logistic Regression
- HTML
- Git and GitHub

## Project Structure

```text
audio-emotion-analysis/
│
├── images/
│   └── my_photo.jpeg
│
├── models/
│   ├── label_encoder.pkl
│   ├── logistic_regression_model.pkl
│   └── tfidf_vectorizer.pkl
│
├── index.html
├── main.py
├── .gitignore
└── README.md
```

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/thv3829/audio-emotion-analysis.git
```

### 2. Navigate to the project directory

```bash
cd audio-emotion-analysis
```

### 3. Create a virtual environment

```bash
python -m venv .venv
```

### 4. Activate the virtual environment

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 5. Install dependencies

Install the Python packages required by the application.

```bash
pip install -r requirements.txt
```

### 6. Run the application

```bash
python main.py
```

Open the local address displayed in the terminal to access the website.

## Project Purpose

This project demonstrates the integration of machine learning models into a web application, combining model inference, data processing, and a user-facing interface.

## Author

Developed as a personal machine learning and web development project.

## Repository

[GitHub Repository](https://github.com/thv3829/audio-emotion-analysis)
