# Box Office 7-Day Domestic Revenue Prediction

This repository contains an end-to-end **machine learning pipeline for predicting first-week (7-day) domestic box office revenue** for theatrical film releases. The project combines large-scale data scraping, feature engineering grounded in real-world box office behavior, and comparative modeling to produce reliable and interpretable forecasts.

---

## Project Objective

The objective is to predict **opening-week domestic revenue** using information available *at or before release*, including:
- Release scale and timing
- Marketing intensity
- Cast, director, and franchise signals
- Distribution and studio backing
- Competitive and seasonal effects

The final pipeline is also used to generate a **forward-looking prediction** for *Avatar: Fire and Ash*.

---

## Data Sources & Files

### Core Datasets

- **`master_reduced_2.csv`**  
  The primary dataset used for model training.  
  It was constructed by scraping and integrating data from:
  - **TMDB (The Movie Database)** – metadata, cast, crew, genres, budgets, languages, collections, marketing assets
  - **BoxOfficeMojo.com** – domestic box office revenue and release information  

- 
- **`Features.csv`**  
  Contains the finalized list of selected features after preprocessing and feature engineering.  
  This file ensures consistency between training, validation, test, and inference pipelines.
**`avatar_fire.csv`**  
  A single-movie inference dataset containing feature values for *Avatar: Fire and Ash*.  
  This file is preprocessed to match the training feature schema and is used exclusively for prediction.

---

## Project Structure

### **Part 1: Preprocessing & Feature Engineering**
Implemented in `Dataset_preprocessing.ipynb`.

Key steps include:
- Cleaning and validating scraped data
- Feature normalization and log transformations
- Marketing intensity aggregation
- Franchise and sequel feature construction
- Holiday and seasonality encoding
- Same-day competition indicators
- Leakage-safe categorical standardization

This stage produces a **model-ready dataset** and the finalized feature list.

---

### **Part 2: Model Training & Prediction**
Implemented in `box_office_model.ipynb`.

Key steps include:
- Time-aware train / validation / test split
- Baseline modeling with **Linear Regression**
- Non-linear modeling with **LightGBM** and **CatBoost**
- Hyperparameter tuning using **Optuna**
- Evaluation in both log space and original revenue scale
- Final inference on *Avatar: Fire and Ash*

---

## Model Performance Summary

| Model | RMSE (log) | R² (log) | MAE (USD) | RMSE (USD) |
|-----|-----------|----------|-----------|------------|
| Linear Regression | 0.9539 | 0.8136 | 16.33M | — |
| LightGBM | 0.7746 | 0.8771 | 10.57M | 28.38M |
| **CatBoost** | **0.7626** | **0.8809** | **10.48M** | **27.66M** |

**CatBoost** was selected as the final model due to its superior accuracy, robustness to overfitting, and native handling of categorical features.

---

## Final Prediction Example

Using the tuned CatBoost model, the predicted **7-day domestic box office revenue** for:

> **Avatar: Fire and Ash**

is: $124,989,450.64

This represents the model’s best point estimate given the assumed release scale, marketing intensity, franchise context, and timing features.

---

## Technical Highlights

- Log-space modeling for heavy-tailed revenue distributions
- Cyclical encoding for seasonal effects
- Franchise-aware feature design
- Ordered boosting to reduce target leakage
- Strict preprocessing parity between training and inference

---

## Tools & Libraries

- Python, Pandas, NumPy
- Scikit-learn
- CatBoost
- LightGBM
- Optuna

---

## Disclaimer

Predictions depend on the **assumptions provided at inference time** (e.g., theater count, marketing assets, release timing). Actual box office performance may differ due to market conditions, audience reception, or unforeseen external factors.

---

## Author
Faiyaz Ahmed
Student at the Institute of Business Administration,University of Dhaka.
Developed as part of a **box office prediction research project**, an academic project for the Course: **Applied Machine Learning for Business** integrating data engineering, econometrics, and applied machine learning for real-world forecasting.



