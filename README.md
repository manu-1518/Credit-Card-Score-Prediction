# 🏦 Credit Score Prediction using Machine Learning

This project predicts whether a loan applicant is a **good or bad credit risk** using classical machine learning models. Built with preprocessing pipelines, model evaluation tools, and a **Streamlit UI for deployment**, it's ideal for real-world credit scoring tasks.

---

## 📌 Table of Contents
- [Overview](#-overview)
- [Dataset](#-dataset)
- [Features](#-features)
- [Modeling](#-modeling)


---

## 🔍 Overview

The goal is to classify loan applicants as **creditworthy (0)** or **non-creditworthy (1)** using various attributes like job type, savings, credit history, age, housing type, and more. The project handles **imbalanced data**, performs **feature encoding**, and tunes models for **F1-score**, focusing on minimizing false negatives (important in loan approvals).

---

## 📊 Dataset

- **Source:** [UCI German Credit Dataset](https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data))
- **Instances:** 1000
- **Features:** 20 attributes + 1 target
- **Target:** 
  - `0` → Good Credit
  - `1` → Bad Credit

---

## 📌 Features

- Preprocessing pipeline with `LabelEncoder` and `StandardScaler`
- Handling imbalanced data using **SMOTE**
- Models: Logistic Regression, XGBoost, CatBoost
- Hyperparameter tuning for threshold optimization
- Modular project structure
- Live demo with **Streamlit UI**

---

## 🧠 Modeling

| Model               | Accuracy | F1 (Bad Class) | Notes                           |
|--------------------|----------|----------------|----------------------------------|
| Logistic Regression| 74%      | 0.56           | Baseline                         |
| XGBoost            | 79%      | 0.64           | Tuned with threshold optimization|
| CatBoost           | 76%      | 0.59           | Good with categorical features   |

---
