# Fraud_Detection_Prediction_with_ML
Fraud Detection Prediction using Machine Learning This project aims to detect fraudulent transactions using machine learning algorithms. It includes data preprocessing, exploratory data analysis, model training, and evaluation using metrics like accuracy, precision, and recall to identify fraud effectively.

Fraud Detection Prediction using Machine Learning
Project Overview

This project presents a machine learning-based approach to detect fraudulent financial transactions. With the increasing volume of digital transactions, identifying fraudulent activities in real time has become a critical task for financial institutions. The project utilizes the Kaggle dataset and implements a Logistic Regression model to classify transactions as fraudulent or genuine.

Table of Contents

Introduction

Dataset Description

Data Preprocessing

Feature Engineering

Exploratory Data Analysis (EDA)

Model Building

Model Evaluation

Results Summary

Future Enhancements

Technologies Used

1. Introduction

Fraud detection plays a vital role in preventing financial losses in online payment systems. Traditional rule-based methods often fail to detect complex or evolving fraud patterns. This project leverages machine learning to build a predictive model that identifies potentially fraudulent transactions based on various transaction attributes.

2. Dataset Description

Source: Kaggle – Credit Card Fraud Detection Dataset

Attributes: Includes transaction type, amount, origin and destination balances, and fraud indicators.

Target Variable: isFraud (1 for fraudulent, 0 for genuine)

3. Data Preprocessing

Handled missing or irrelevant data fields.

Dropped the step column after analyzing time-based trends.

Applied standard scaling to numerical features using StandardScaler.

Encoded categorical variables using OneHotEncoder.

4. Feature Engineering

Created new features:

balanceDiffOrig = oldbalanceOrg - newbalanceOrig

balanceDiffDest = oldbalanceDest - newbalanceDest

These features capture the difference in balances before and after transactions, helping the model detect unusual money movements.

5. Exploratory Data Analysis (EDA)

Visualized transaction amount distribution on a logarithmic scale to manage skewness.

Plotted fraud occurrences across transaction types and over time.

Boxplot analysis indicated that fraudulent transactions generally involve higher amounts.

Count plots revealed that most frauds occur in TRANSFER and CASH_OUT transactions.

6. Model Building

Used Logistic Regression within a Pipeline for streamlined preprocessing and classification.

Configured the model with class_weight='balanced' to address class imbalance.

Split the data into training and testing sets to evaluate generalization performance.

7. Model Evaluation

Evaluated using Accuracy, Precision, Recall, F1-score, and Confusion Matrix.

Achieved approximately 95% accuracy with a high recall score, ensuring minimal false negatives.

The model effectively distinguishes fraudulent transactions from genuine ones.

8. Results Summary

Fraudulent transactions primarily occur in TRANSFER and CASH_OUT types.

Fraudulent activities generally involve higher transaction amounts.

The Logistic Regression model demonstrates strong predictive performance with reliable recall.

9. Future Enhancements

Extend the model to advanced algorithms such as Random Forest, Gradient Boosting, or XGBoost.

Implement real-time fraud detection using live transaction streams.

Apply hyperparameter tuning and cross-validation for optimization.

Integrate the model into a simple web or API-based interface for real-time prediction.

10. Technologies Used

Programming Language: Python

Libraries: NumPy, Pandas, Scikit-learn, Matplotlib, Seaborn, Joblib

Algorithm: Logistic Regression

Dataset Source: Kaggle

Conclusion

This project demonstrates how machine learning can effectively identify fraudulent transactions using a data-driven approach. Through systematic preprocessing, feature engineering, and evaluation, the Logistic Regression model achieved high performance and provides a strong foundation for further development into a real-time fraud detection system.
