# Life Expectancy Prediction Using WHO Global Health Data

## Project Overview

This project aims to build an explainable supervised learning model that predicts **Life Expectancy** based on various health indicators collected from 193 countries by the **World Health Organization (WHO)** in 2015. The model explores different machine learning techniques and attempts to provide insights into the factors that influence life expectancy.

The project was carried out as part of the **CST4050 - Component 2** coursework and includes detailed steps for data loading, preprocessing, model training, tuning, and interpretation. The final model provides a robust prediction of life expectancy with an R² score of **82%** on unseen data.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Objective](#objective)
- [Methodology](#methodology)
- [Technologies Used](#technologies-used)
- [Results](#results)

## Dataset

The dataset, `supervised.csv`, contains various health indicators for 193 countries, including:
- **Life Expectancy (Target)**: The dependent variable we aim to predict.
- **Health Indicators (Features)**: Health-related attributes such as BMI, GDP, schooling, status (Developed/Developing), immunization rates, mortality rates, and more.

The dataset has missing values for some features, which are addressed in the data preprocessing steps.

## Objective

The primary objective is to build a regression model that predicts **Life Expectancy** using the available health indicators. The model aims to:
- Handle missing data efficiently.
- Ensure the features are properly standardized and transformed.
- Minimize overfitting while maximizing model accuracy.
- Interpret which features contribute most to the prediction.

## Methodology

1. **Data Loading and Preliminary Analysis**:
   - Loaded the dataset using Pandas.
   - Analyzed the target (`Life Expectancy`) for skewness and distribution.
   - Investigated feature relationships and missing values using correlation heatmaps and visualizations.

2. **Feature Engineering**:
   - Removed features with high cardinality and excessive missing values.
   - Used mode imputation for skewed features with less than 10% missing values.
   - Coded categorical variables (e.g., `Status`: Developed/Developing).

3. **Model Training and Testing**:
   - Created a machine learning pipeline using **Linear Regression** and **StandardScaler**.
   - Applied **K-Fold Cross Validation** (5 splits) to ensure robustness.
   - Evaluated model accuracy using the R² score.

4. **Tuning the Model**:
   - Addressed overfitting by implementing **Ridge Regression** (L2 Regularization).
   - Tuned the regularization parameter `alpha` using a range of values with cross-validation to find the best performing model.

5. **Model Interpretation**:
   - Interpreted feature importance by examining the Ridge Regression coefficients.
   - Explored multicollinearity and its impact on prediction.

6. **Discussion**:
   - Time complexity of the pipeline was analyzed.
   - Discussed pros/cons and explored possible improvements to the pipeline.


## Technologies Used
- **Python**: Core programming language for the project.
- **Pandas, NumPy**: Data manipulation and analysis.
- **Scikit-learn**: Machine learning models and evaluation metrics.
- **Seaborn, Matplotlib**: Data visualization.
- **Ridge Regression, Linear Regression**: Models used for prediction.

## Results

- **R² Score on Train Set**: 88%
- **R² Score on Test Set**: 54% (Initial Linear Regression model)
- **Final Tuned Ridge Regression**:
  - **Train Set Accuracy**: 82%
  - **Test Set Accuracy**: 82%
- The final model demonstrated a significant improvement in performance by addressing overfitting and selecting the best regularization parameter.


