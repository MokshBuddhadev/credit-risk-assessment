# Credit Risk Assessment using Machine Learning

## Overview

This project implements a complete machine learning pipeline for credit risk assessment, where the goal is to predict whether a customer will default on a loan based on financial and payment history data.

Multiple supervised, ensemble, boosting, deep learning, and unsupervised learning algorithms are compared to select the best performing model using evaluation metrics such as ROC-AUC, F1-score, Precision, Recall, and Accuracy.

This project follows a real-world workflow used in banking, fintech, and risk analytics systems.

---

## Problem Statement

Financial institutions must evaluate the risk before approving loans.  
Incorrect decisions can lead to financial loss.

Goal:

Predict whether a customer will default on payment next month.

Output:

0 → No Default  
1 → Default  

This is a supervised binary classification problem.

---

## Dataset

Dataset used: UCI Credit Card Default Dataset

Features include:

- Credit limit
- Payment history
- Bill amounts
- Previous payments
- Demographic information

Target variable:

default payment next month

Dataset size:

- 30,000 records
- 23 features

---

## Project Pipeline

1. Data Loading  
2. Data Preprocessing  
3. Train-Test Split  
4. Feature Scaling  
5. Handling Imbalanced Data using SMOTE  
6. Model Training  
7. Model Comparison  
8. Feature Importance Analysis  
9. Unsupervised Learning (Clustering)  
10. Final Model Selection  

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Matplotlib
- Imbalanced-learn

---

## Data Preprocessing

Steps performed:

- Renamed target column
- Train-test split
- Standard scaling
- SMOTE for class imbalance

Why SMOTE?

The dataset is imbalanced, meaning there are more non-default cases than default cases.  
SMOTE generates synthetic samples to balance the dataset.

---

## Models Implemented

Supervised Models

- Logistic Regression
- Decision Tree
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- Naive Bayes

Ensemble Models

- Random Forest
- Gradient Boosting
- AdaBoost

Boosting Model

- XGBoost

Deep Learning

- Neural Network (MLPClassifier)

Unsupervised Learning

- KMeans Clustering
- PCA Visualization

---

## Evaluation Metrics

Models were evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC Score
- Silhouette Score (for clustering)

Why multiple metrics?

Credit risk datasets are imbalanced, so accuracy alone is not enough.  
Recall is important because missing a risky customer can cause financial loss.

---

## Model Performance Summary

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|--------|----------|------------|--------|-----|----------|
| Logistic Regression | 0.78 | 0.63 | 0.49 | 0.55 | 0.76 |
| Decision Tree | 0.79 | 0.65 | 0.54 | 0.59 | 0.77 |
| Random Forest | 0.82 | 0.71 | 0.60 | 0.65 | 0.83 |
| Gradient Boosting | 0.83 | 0.72 | 0.61 | 0.66 | 0.84 |
| AdaBoost | 0.81 | 0.69 | 0.58 | 0.63 | 0.82 |
| SVM | 0.82 | 0.70 | 0.59 | 0.64 | 0.83 |
| KNN | 0.80 | 0.66 | 0.55 | 0.60 | 0.79 |
| Naive Bayes | 0.76 | 0.60 | 0.47 | 0.53 | 0.74 |
| XGBoost | 0.85 | 0.74 | 0.66 | 0.70 | 0.87 |
| Neural Network | 0.84 | 0.73 | 0.64 | 0.68 | 0.86 |

---

## Best Model Selected

Final Model: XGBoost Classifier

Reasons:

- Highest ROC-AUC score
- Highest F1 score
- Good recall on risky customers
- Works well on tabular financial data
- Handles non-linear relationships

Improvement over baseline:

| Metric | Logistic | XGBoost | Improvement |
|---------|----------|----------|------------|
| Accuracy | 0.78 | 0.85 | +7% |
| Recall | 0.49 | 0.66 | +17% |
| F1 Score | 0.55 | 0.70 | +15% |
| ROC-AUC | 0.76 | 0.87 | +11% |

---

## Feature Importance

Random Forest was used to identify the most important features.

Important factors:

- Payment history
- Bill amount
- Credit limit
- Previous delay

This shows the model makes realistic decisions similar to real banking systems.

---

## Unsupervised Learning

KMeans clustering was used to identify hidden patterns in customers.

Purpose:

- Customer segmentation
- Risk grouping
- Exploratory analysis

PCA was used for visualization.

This approach is useful in real-world risk analytics.

---

## Real World Applications

- Banking loan approval systems
- Credit card risk scoring
- Fintech risk engines
- Fraud detection systems
- Financial analytics platforms

Even a small improvement in recall can reduce financial loss significantly.

---

## How to Run

Install dependencies

pip install pandas numpy scikit-learn xgboost imbalanced-learn matplotlib

Run notebook

02_modeling.ipynb

---

## Future Improvements

- Hyperparameter tuning
- Cross validation
- Model deployment
- Explainable AI (SHAP, LIME)
- Web dashboard

---

## Author

Moksh Buddhadev
