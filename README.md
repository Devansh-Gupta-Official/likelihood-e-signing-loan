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

## Step 3: Data Preprocessing
### Feature Engineering
The months_employed column is identified as potentially inaccurate and is subsequently dropped from the dataset. Additionally, two columns (personal_account_m and personal_account_y) are combined into a new feature, personal_account_months, to simplify the data structure.

### One Hot Encoding
Categorical variables are encoded using one-hot encoding to facilitate the model training process.

### Removing Extra Columns
Columns such as entry_id and e_signed are dropped as they do not contribute to the predictive modeling.

### Train-Test Split
The dataset is split into training and testing sets using the train_test_split function from scikit-learn.

### Feature Scaling
Standard scaling is applied to standardize the numerical features, ensuring that they have a mean of 0 and a standard deviation of 1.

## Step 4: Model Building
### Comparing Models
Logistic Regression, Support Vector Machines (SVM) with linear and radial basis function (RBF) kernels, and Random Forest models are trained and evaluated. The evaluation metrics include accuracy, precision, recall, and F1 score.

### K-Fold Cross Validation
K-Fold cross-validation is performed to assess the models' performance across multiple folds.

## Step 5: Model Improvement
### Grid Search
Grid search is utilized to find optimal hyperparameters for the Random Forest model. Two rounds of grid search are conducted, first using entropy as the criterion and then using the Gini index.

### Testing New Parameters on Test Set
The best-performing Random Forest models from both grid search rounds are evaluated on the test set, and the results are compared.

## Step 6: Saving and Finalizing Results
The model evaluation results are saved to a CSV file (model_results.csv). The final predictions on the test set, along with user identifiers, are saved to another CSV file (final_results.csv).

This notebook serves as a comprehensive guide to understanding the entire process of building, evaluating, and improving a predictive model for e-signing loans. The steps include data preprocessing, exploratory data analysis, feature engineering, model building, and hyperparameter tuning.
