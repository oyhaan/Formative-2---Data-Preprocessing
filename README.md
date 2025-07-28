# Formative-2---Data-Preprocessing

# User Identity and Product Recommendation System

This repository contains the implementation of a multimodal authentication and product recommendation system. The system uses **facial recognition**, **voice verification**, and **customer data** to provide personalized product recommendations to verified users.

## 🚀 Project Overview

The system workflow includes:
1. **Facial Recognition** – Identifies if the user is in the known dataset.
2. **Voiceprint Verification** – Confirms the user’s intent to proceed with the transaction.
3. **Product Recommendation Model** – Predicts the product the verified user is most likely to purchase.

All steps must succeed for the recommendation to be displayed. Unauthorized access is blocked at each step.

---

## 🛠️ Features

- Merged tabular customer data from two sources
- Image preprocessing, augmentation, and feature extraction
- Audio preprocessing, augmentation, and feature extraction
- Trained models using Random Forest, Logistic Regression, and/or XGBoost
- CLI simulation of full system flow
- Unauthorized access test cases

---

## 📊 Performance Metrics

Each model is evaluated using:
- Accuracy
- F1-Score
- Loss

---

## 🎥 Deliverables

- ✅ Final Report (PDF)
- ✅ Simulation Video Link
- ✅ GitHub Repository (this)
- ✅ Jupyter Notebook for analysis
- ✅ CLI simulation script

---

## 👥 Team Members

- **Abiodun Israel Kumuyi**
- **Anissa Tegawendé Ouedraogo**
- **Gaius Irakiza**
- **Ganza Owen Yhaan**
- **Nicolas Muhigi**

---

## 👥 Team Contributions

Each team member contributed:
- 3 facial images (neutral, smiling, surprised)
- 2 audio clips (e.g., “Yes, approve”, “Confirm transaction”)
- Assisted with model development and system simulation

---

## 🔧 Requirements

Install dependencies:
```bash
pip install -r requirements.txt

