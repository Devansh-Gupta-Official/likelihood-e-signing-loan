#  Likelihood of E-Signing a Loan Analysis

## Overview
This Jupyter Notebook (likelihood.ipynb) focuses on analyzing a dataset related to e-signing of loans. The dataset (P39-Financial-Data.csv) includes various features such as age, pay schedule, home ownership status, income, risk scores, and the target variable indicating whether the loan was e-signed. The goal is to build and evaluate machine learning models to predict the likelihood of e-signing a loan. The notebook primarily covers data preprocessing, exploratory data analysis, feature engineering, model building, and evaluation.

## Step 1: Importing Libraries and Data
The initial section involves importing essential Python libraries such as Pandas, NumPy, Matplotlib, and Seaborn. The dataset (P39-Financial-Data.csv) is loaded into a Pandas DataFrame (df).

## Step 3: Data Visualization
### Cleaning the Data
Descriptive statistics of the dataset are presented using the describe() method. Additionally, the presence of any missing values is checked using the isna().any() method, indicating that there are no missing values.

### Histograms
Histograms of numerical columns are plotted to visualize the distribution of data, providing insights into the characteristics of different features.

### Correlation Analysis
- **Correlation Plot (with response variable)**
A bar chart is created to display the correlation of each feature with the response variable (e_signed). This helps understand the linear relationship between features and the target variable.

- **Correlation Matrix**
A heatmap visualizes the correlation matrix, allowing for a comprehensive view of the pairwise relationships between features.

