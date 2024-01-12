#  Likelihood of E-Signing a Loan Analysis

## Overview
This Jupyter Notebook (likelihood.ipynb) focuses on analyzing a dataset related to e-signing of loans. The dataset (P39-Financial-Data.csv) includes various features such as age, pay schedule, home ownership status, income, risk scores, and the target variable indicating whether the loan was e-signed. The goal is to build and evaluate machine learning models to predict the likelihood of e-signing a loan. The notebook primarily covers data preprocessing, exploratory data analysis, feature engineering, model building, and evaluation.

## Step 1: Importing Libraries and Data
The initial section involves importing essential Python libraries such as Pandas, NumPy, Matplotlib, and Seaborn. The dataset (P39-Financial-Data.csv) is loaded into a Pandas DataFrame (df).

## Step 2: Data Visualization
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
- **Logistic Regression**
Logistic Regression is used as a baseline model. The LogisticRegression class from scikit-learn is employed, with L1 regularization (Lasso) to potentially select important features. The model is trained on the training set and evaluated on the test set using metrics such as accuracy, precision, recall, and F1 score.

- **Support Vector Machines (SVM)**
**1. SVM with Linear Kernel**
A Support Vector Machine (SVM) with a linear kernel is implemented using the SVC class from scikit-learn. The model is trained and evaluated similarly to the Logistic Regression model, and performance metrics are recorded.

**2. SVM with Radial Basis Function (RBF) Kernel**
Another SVM model is created, this time using an RBF kernel. The same training and evaluation process is followed, and the model's performance is compared with the linear SVM.

- **Random Forest Model**
A Random Forest classifier is employed for more complex, ensemble-based modeling. The RandomForestClassifier class from scikit-learn is used with 100 decision trees and entropy as the criterion. The model is trained on the training set and evaluated on the test set, recording performance metrics.

### K-Fold Cross Validation
K-Fold cross-validation is performed to assess the models' performance across multiple folds.

## Step 5: Model Improvement
### Grid Search for Random Forest
- **Round 1: Entropy**
Grid search is conducted to find the optimal hyperparameters for the Random Forest model. Various hyperparameters, such as max_depth, max_features, min_samples_split, min_samples_leaf, bootstrap, and criterion (set to entropy), are explored using the GridSearchCV class from scikit-learn. The best-performing set of hyperparameters is recorded.

- **Round 2: Gini**
A second round of grid search is performed, this time using Gini as the criterion. The same hyperparameters are explored, and the optimal set is identified.

### Testing New Parameters on Test Set
The best-performing Random Forest models from both rounds of grid search are evaluated on the test set. The results are compared to determine the impact of the hyperparameter tuning on the model's performance.

## Step 6: Saving and Finalizing Results
The model evaluation results are saved to a CSV file (model_results.csv). The final predictions on the test set, along with user identifiers, are saved to another CSV file (final_results.csv).

This notebook serves as a comprehensive guide to understanding the entire process of building, evaluating, and improving a predictive model for e-signing loans. The steps include data preprocessing, exploratory data analysis, feature engineering, model building, and hyperparameter tuning.
