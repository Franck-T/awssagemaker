# 📱 Mobile Price Range Classification With AWS SageMaker
### Machine Learning • Classification • Model Training • EDA


## 📌 Overview

This project builds a **machine learning model that predicts the price range of a mobile device** based on its hardware specifications.  
It is a supervised classification problem where the model predicts one of multiple price categories:

- **0 — Low Cost**  
- **1 — Medium Cost**  
- **2 — High Cost**  
- **3 — Very High Cost**

This project demonstrates:

- Exploratory Data Analysis (EDA)  
- Feature engineering  
- Model building & comparison  
- Performance evaluation  
- Realistic classification workflow for business decision-making
- Deploy-ready on AWS Sagemaker  


## 🎯 Problem Statement

Mobile companies want to understand how hardware specs influence device pricing.  
Given details such as:

- Battery power  
- RAM  
- Weight  
- Screen resolution  
- CPU cores  
- Storage capacity  
- Camera resolution  

We want to **predict the price category** of a mobile phone.

This helps in:

- Pricing optimization  
- Market segmentation  
- Product planning  
- Competitive benchmarking  


## 📂 Dataset Description

Your dataset (mob_price_classification_train.csv) includes features such as:

- `battery_power` – battery capacity in mAh  
- `ram` – device RAM  
- `mobile_wt` – weight of the device  
- `px_height`, `px_width` – screen resolution  
- `cores` – number of CPU cores  
- `talk_time` – maximum talk time  
- `int_memory` – internal memory  
- `pc` – primary camera MP  
- `fc` – front camera MP  
- `blue`, `wifi`, `dual_sim`, `4g`, `3g` – boolean flags  
- `price_range` – the target variable (0 = cheap, 3 = expensive)


## Data Transformation

The dataset is split into train and test set and stored as .csv files which will be uploaded to an S3 bucket to be used for training.

## 🤖 Machine Learning Models

Models evaluated:

- Random Forest  

Metrics used:

- Accuracy  
- Precision  
- Recall  
- F1-score  

## 🧰 Tech Stack

- Python  
- Pandas  
- NumPy  
- Matplotlib  
- Seaborn  
- Scikit-learn  
- Random Forest 
- Jupyter Notebook  
- AWS SageMaker

## 🚀 Results Summary

```
---- METRICS RESULTS FOR TESTING DATA ----
Total Rows are:  300
[TESTING] Model Accuracy is:  0.8833333333333333
[TESTING] Testing Report: 
              precision    recall  f1-score   support
           0       0.95      1.00      0.97        69
           1       0.85      0.80      0.83        66
           2       0.80      0.77      0.79        74
           3       0.91      0.95      0.93        91
    accuracy                           0.88       300
   macro avg       0.88      0.88      0.88       300
weighted avg       0.88      0.88      0.88       300

```
- The Random Forest Classifier performed as shown below:
- Achieved high accuracy in predicting price category
- Model generalizes well on unseen data


