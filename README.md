# Automated Content Moderation System


https://github.com/user-attachments/assets/5aa93360-d553-4d54-89b7-44221c80bd03


## Introduction

This project implements an automated content moderation system that classifies textual content as either **Appropriate** or **Inappropriate** for public display. Leveraging advanced machine learning models (LSTM and BERT) and natural language processing techniques, the system analyzes textual data in real-time to provide accurate classification results. Additionally, the project extends its functionality to handle image inputs—extracting content from images and memes and classifying them using a Hugging Face NSFW image detection model.

---

## Files Included

- **SAFETY TEST(LSTM).ipynb:**  
  Jupyter notebook for training and evaluating the LSTM model for text classification using the preprocessed data.

- **BERT.ipynb:**  
  Jupyter notebook for training and evaluating the BERT model for text classification using the preprocessed data.

- **Flask_app/main.py:**  
  Python file for the Flask application. This file handles both text and image inputs through a web browser, loads the trained BERT model, and leverages a Hugging Face NSFW image detection model for image classification.

- **Flask_app/templates/index5.html:**  
  HTML page for the Flask application's user interface.

- **Flask_app/app.yaml:**  
  Configuration file required for deploying the Flask application on Google App Engine.

---
Note:

The trained BERT model files are not included in the repository; they need to be trained separately using the provided notebooks.
Make sure to adjust file paths in the Python files according to your local setup.
The image classification now uses a Hugging Face model for NSFW detection.

Results
The performance of our text classification models was evaluated using key metrics such as accuracy, precision, recall, and F1 score. The evaluation was conducted on a test set consisting of 4,957 samples (835 labeled as Appropriate and 4,122 labeled as Inappropriate).

Evaluation Metrics
Metric	LSTM	BERT
Accuracy	93.65%	95.62%
Precision	95.97%	97.20%
Recall	96.41%	97.55%
F1 Score	96.19%	97.37%
These results illustrate that both models perform strongly, with the BERT-based model achieving slightly higher metrics overall.
