# Customer Churn Prediction

A machine learning project to predict customer churn using data mining techniques and classification models. This project demonstrates the complete workflow from data preprocessing to model evaluation and feature importance analysis.

**Course:** MSAI 535 001 - Data Mining and Knowledge Discovery  
**Instructor:** HAS Sothea, PhD  
**Student:** Dara NEB (ID: 2026024)

---

## Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Results](#results)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Key Findings](#key-findings)
- [References](#references)

---

## Overview

This project builds a binary classification model to predict whether a customer will leave (churn) or stay with a company. By identifying at-risk customers early, businesses can implement targeted retention strategies and improve customer lifetime value.

The project demonstrates a complete data science workflow including:
- Data cleaning and preprocessing
- Exploratory data analysis (EDA)
- Feature engineering and transformation
- Baseline and advanced model development
- Hyperparameter tuning using randomized search
- Model evaluation and comparison
- Feature importance analysis

---

## Problem Statement

### Objective

Develop a predictive model to classify customers as likely to churn or not, based on their characteristics and service usage patterns.

### Key Questions

1. Which customer factors are most strongly associated with churn?
2. Can we accurately predict churn before it happens?
3. How can businesses prioritize retention efforts?

### Target Variable

- **Churn (Binary):** Yes (customer leaves) or No (customer stays)

---

## Dataset

### Source

Kaggle customer churn dataset for a telecommunications company.

### Size

- **Training set:** 7,043 customer records
- **Test set:** 2,974 customer records

### Features

#### Demographic Information
- `gender`: Male or Female
- `SeniorCitizen`: Binary indicator (0 or 1)
- `Partner`: Whether customer has a partner
- `Dependents`: Whether customer has dependents

#### Account Information
- `tenure`: Number of months as a customer
- `Contract`: Type of contract (Month-to-month, One year, Two year)
- `PaperlessBilling`: Whether customer uses paperless billing
- `PaymentMethod`: Payment method used

#### Service Usage
- `PhoneService`: Whether customer has phone service
- `MultipleLines`: Whether customer has multiple phone lines
- `InternetService`: Type of internet service (Fiber optic, DSL, No)
- `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`: Support services
- `StreamingTV`, `StreamingMovies`: Streaming services

#### Billing Information
- `MonthlyCharges`: Monthly charge to the account
- `TotalCharges`: Total amount charged to the customer

---

## Methodology

### 1. Data Preprocessing

**Data Cleaning:**
- Removed non-informative features (e.g., customer ID)
- Corrected data types (e.g., `SeniorCitizen` as categorical)
- Verified no missing values or duplicate records
- Identified and handled outliers in numerical features

**Feature Consolidation:**
- Merged `No internet service` and `No phone service` values into `No` category
- Reduced unnecessary categories to improve model efficiency

**Feature Transformation:**
- Binary encoding: Converted yes/no values to 1/0
- One-hot encoding: Expanded categorical variables (`InternetService`, `Contract`, `PaymentMethod`)
- Feature engineering: Created `AvgMonthly` and `Monthly_minus_AvgMonthly` features

**Data Alignment:**
- Ensured train and test sets have identical feature sets

### 2. Exploratory Data Analysis (EDA)

**Analyses Performed:**
- Value counts for categorical features
- Distribution analysis for numerical features (histograms with KDE)
- Correlation matrix to identify relationships
- Box plots to compare numerical features between churn and non-churn groups
- Outlier detection and treatment

**Key Insights:**
- Churned customers tend to have shorter tenure and higher monthly charges
- Contract type and service bundling are strongly related to churn
- Outliers in tenure, total charges, and monthly charges were capped to reduce noise

### 3. Baseline Model

**Model:** Logistic Regression with L2 regularization

**Performance:**
- Validation Accuracy: 0.84
- Scaled features using StandardScaler

### 4. Advanced Model: XGBoost Classifier

**Initial XGBoost:**
- n_estimators=100, learning_rate=0.1, max_depth=5
- Validation Accuracy: 0.8898

### 5. Hyperparameter Tuning

**Method:** RandomizedSearchCV with StratifiedKFold CV (3 splits)

**Hyperparameter Search Space:**
- `n_estimators`: [3000, 5000, 8000, 10000]
- `learning_rate`: [0.005, 0.01, 0.015, 0.02, 0.03]
- `max_depth`: [3, 4, 5, 6, 8, 10]
- `min_child_weight`, `subsample`, `colsample_bytree`: Various values
- `gamma`, `reg_alpha`, `reg_lambda`: Regularization parameters
- `scale_pos_weight`: Class weight adjustment

**Evaluation Metric:** ROC-AUC (chosen for its ability to rank customers by churn risk)

---

## Results

### Final Model Performance

**Tuned XGBoost (Selected Final Model)**

| Metric | Value |
|--------|-------|
| Accuracy | 0.8690 |
| ROC-AUC | 0.9563 |
| Precision | High performance across classes |
| Recall | Strong identification of churn cases |

### Model Comparison

| Model | Validation Accuracy | Validation ROC-AUC | Selection Rationale |
|-------|---:|---:|---|
| Baseline (Logistic Regression) | 0.84 | — | Starting benchmark |
| Regular XGBoost | 0.8898 | — | Good tree-based performance |
| **Tuned XGBoost** | **0.8690** | **0.9563** | **Selected - Best ROC-AUC** |
| Top-5-Feature XGBoost | 0.7914 | — | Simpler, interpretable model |

**Why Tuned XGBoost?**
- Highest ROC-AUC (0.9563) indicates superior ability to rank customers by churn risk
- Strong validation accuracy despite slightly lower than baseline
- Cross-validated performance ensures generalization
- Provides feature importance for business insights

### Top Features for Churn Prediction

The tuned XGBoost model identifies the following as the most important predictors:
1. **Contract Type** - Strong predictor of stability
2. **Tenure** - Customer loyalty duration
3. **Total Charges** - Cumulative customer value
4. **Monthly Charges** - Price sensitivity
5. Additional service-related features

A simplified model using only these top 5 features achieved 0.7914 accuracy, showing that these factors are highly informative.

---

## Project Structure

```
customer-churn-prediction/
├── Customer_Churn_Prediction.ipynb    # Main analysis notebook
├── README.md                           # This file
├── train.csv                          # Training dataset (external)
└── test.csv                           # Test dataset (external)
```

### Notebook Sections

1. **Introduction** - Project context and motivation
2. **Data Preprocessing** - Cleaning and transformation
3. **Exploratory Data Analysis** - Pattern discovery
4. **Data Mining Techniques** - Model development and tuning
5. **Results and Evaluation** - Performance comparison
6. **Discussion and Challenges** - Key learnings
7. **Conclusion and Future Work** - Recommendations

---

## Installation

### Requirements

- Python 3.8+
- Jupyter Notebook

### Dependencies

```bash
pip install pandas matplotlib seaborn scikit-learn xgboost numpy
```

Or install all at once:

```bash
pip install pandas==2.0+ matplotlib seaborn scikit-learn==1.3+ xgboost numpy
```

---

## Usage

### Running the Notebook

1. **Clone the repository:**
   ```bash
   git clone https://github.com/vibecoder1998/customer-churn-prediction.git
   cd customer-churn-prediction
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   # or use the pip install command above
   ```

3. **Prepare the data:**
   - Place `train.csv` and `test.csv` in the working directory or update paths in the notebook
   - Expected location: `/Users/lonwolf/Desktop/Data Mining/`

4. **Open Jupyter Notebook:**
   ```bash
   jupyter notebook Customer_Churn_Prediction.ipynb
   ```

5. **Run cells sequentially:**
   - Cell 1-5: Setup and data loading
   - Cell 6-15: Data cleaning and preprocessing
   - Cell 16-25: Exploratory data analysis
   - Cell 26-35: Baseline and XGBoost models
   - Cell 36-40: Hyperparameter tuning
   - Cell 41-45: Feature importance and top-5 model evaluation

### Making Predictions

The tuned XGBoost model is available as `best_xgb` (from `random_search.best_estimator_`):

```python
# Single prediction
prediction = best_xgb.predict(X_new)

# Probability of churn (0-1)
churn_probability = best_xgb.predict_proba(X_new)[:, 1]
```

---

## Key Findings

### Churn Patterns

1. **Tenure Effect:** New customers (< 3 months) have significantly higher churn rates
2. **Contract Type:** Month-to-month customers are more likely to churn than those with longer contracts
3. **Billing Behavior:** Customers with higher monthly charges but lower tenure are at higher risk
4. **Service Bundling:** Customers with more services (security, backup, support) tend to stay longer

### Business Implications

- **Target:** Focus retention efforts on customers in critical time windows (first 3 months)
- **Strategy:** Encourage longer contracts and service bundling
- **Pricing:** Balance pricing strategy to avoid losing price-sensitive customers
- **Support:** Proactive support for high-risk customers can improve retention

### Model Limitations

- Default prediction threshold may need adjustment based on business costs
- SHAP values could provide deeper insights into individual predictions
- Model performance may vary with evolving customer behavior
- Regular retraining recommended as data patterns change

---

## Discussion and Challenges

### Key Challenges Addressed

1. **Data Quality:**
   - Simplified confusing categories (`No internet service` → `No`)
   - Handled outliers through capping rather than removal

2. **Metric Selection:**
   - Chose ROC-AUC over accuracy due to importance of ranking customers by risk
   - Stratified cross-validation ensures balanced class representation

3. **Feature Engineering:**
   - Created relevant features (`AvgMonthly`) to help tree splits
   - One-hot encoding for categorical variables

### Observations

- Tree-based models (XGBoost) outperform linear models (Logistic Regression)
- Hyperparameter tuning provides substantial improvements
- Feature importance helps identify actionable business factors

---

## References

- **XGBoost:** Chen, T., and Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*.

- **Scikit-learn:** Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825-2830.

- **Project Reference:** [customer-churn-prediction](https://github.com/vibecoder1998/customer-churn-prediction)

- **Kaggle Datasets:** https://www.kaggle.com/

- **Documentation:**
  - [XGBoost Documentation](https://xgboost.readthedocs.io/)
  - [Scikit-learn Documentation](https://scikit-learn.org/)
  - [Pandas Documentation](https://pandas.pydata.org/)
  - [Matplotlib Documentation](https://matplotlib.org/)
  - [Seaborn Documentation](https://seaborn.pydata.org/)

---

## Author

**Dara NEB**  
Student ID: 2026024  
Email: 2026024neb@aupp.edu.kh  
Course: MSAI 535 001 - Data Mining and Knowledge Discovery

---

## License

This project is part of academic coursework at AUPP. Please contact the author for usage permissions.

---
