
# 🛍️ E-commerce Customer Churn Prediction

Welcome to the **E-commerce Customer Churn Prediction** project!  
This project uses real-world transactional data from a Brazilian e-commerce platform to **predict which customers are likely to churn** — and explains the reasoning behind each prediction using SHAP values.

---

##  Project Overview

In any online business, retaining customers is just as important (if not more) than acquiring new ones. Using machine learning, this project helps:

- **Identify at-risk customers** before they churn.
- **Understand the drivers** behind customer churn.
- **Visualize insights** that could help teams improve retention strategies.

---

## 📂 Project Structure
E-commerce-Churn-Prediction/
│
├── notebooks/
│ ├── 01_data_understanding.ipynb <- Load and explore raw datasets
│ ├── 02_merge&prepare.ipynb <- Merge and clean all datasets
│ ├── 03_feature_engineering.ipynb <- Create churn labels and features
│ ├── 04_model_training.ipynb <- Build and evaluate ML model
│ ├── 05_shap_explainability.ipynb <- Visualize SHAP explanations
│
├── outputs/
│ ├── churn_model.pkl <- Trained Random Forest model
│ ├── X.pkl / y.pkl <- Final features and target
│
├── Streamlit__app.py <- Streamlit app for interactive prediction
├── requirements.txt <- Required Python packages
└── README.md <- Project overview (this file)


---

##  Datasets Used

All data comes from the [Olist Brazilian E-commerce Dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce).  
It contains real customer orders, payments, reviews, product details, and more.

Key datasets:
- `orders.csv` - Order-level data with timestamps
- `customers.csv` - Customer IDs and zip codes
- `order_items.csv` - Product-level breakdown
- `order_reviews.csv` - Customer review scores
- `order_payments.csv` - Payment methods and amounts
- `product_category_translation.csv` - Portuguese-to-English mapping

---

## What We Did

### 1. Data Merging & Cleaning
- Combined 6+ raw CSVs into one customer-level dataset.
- Filtered only “delivered” orders.
- Filled missing values and handled outliers.

### 2. Feature Engineering
- Defined **churn** as customers who didn’t return after their last purchase.
- Created features like:
  - Total spend
  - Total orders
  - Average days between purchases
  - Average review score

### 3. Model Training
- Trained a **Random Forest Classifier** to predict churn.
- Achieved high accuracy and good generalization on test data.
- Saved the model and feature matrix as `.pkl` files for reuse.

### 4. Explainability with SHAP
- Used **SHAP (SHapley Additive Explanations)** to explain each prediction.
- Visualized which features pushed the model toward predicting churn vs. retention.

---

## Streamlit App

We built a full **interactive web app** using Streamlit that allows:

- Selecting any customer index to view their prediction
- Seeing their churn probability
- Visualizing **why** the model made that prediction (via SHAP waterfall plot)
- Optionally uploading external customer data for batch predictions

Try it by running:

```bash
streamlit run Streamlit_app.py
