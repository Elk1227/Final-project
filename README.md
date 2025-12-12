# Telecom Customer Churn Prediction

This project builds a machine learning pipeline to predict customer churn in the telecommunications domain using the IBM Telco Customer Churn dataset. The goal is to identify customers who are likely to leave the service, enabling proactive retention strategies.

## Dataset
- **Source**: IBM Telco Customer Churn Dataset  
- **Size**: 7,043 customers  
- **Target Variable**: `Churn Value` (0 = stay, 1 = churn)  
- **Features**: Demographics, service usage, contract type, billing information  

The dataset exhibits class imbalance (~26% churners) and contains many categorical variables with missing values.

## Methodology
The project implements a leakage-free machine learning pipeline using `scikit-learn` and `imbalanced-learn`, including:
- Missing value imputation
- One-hot encoding (nominal features)
- Ordinal encoding (contract type)
- Feature scaling
- SMOTE oversampling to address class imbalance

Five models are evaluated:
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- Random Forest
- XGBoost

Hyperparameter tuning is performed for Random Forest and XGBoost using `GridSearchCV`.

## Results
XGBoost achieves the best performance with approximately:
- **Accuracy**: 0.80  
- **ROC-AUC**: 0.85–0.86  

Results are evaluated on a held-out test set using ROC-AUC as the primary metric.

## How to Run
1. Clone this repository:
```bash
git clone https://github.com/Elk1227/Final-project.git
cd Final-project
```
2.Install dependencies:
```bash
pip install -r requirements.txt
```
3.Run the main script:
```bash
python main.py
```
