#  Likelihood of E-Signing a Loan Analysis

## Overview
This Jupyter Notebook (likelihood.ipynb) focuses on analyzing a dataset related to e-signing of loans. The dataset (P39-Financial-Data.csv) includes various features such as age, pay schedule, home ownership status, income, risk scores, and the target variable indicating whether the loan was e-signed. The goal is to build and evaluate machine learning models to predict the likelihood of e-signing a loan. The notebook primarily covers data preprocessing, exploratory data analysis, feature engineering, model building, and evaluation.

Leading companies work by analyzing the financial history of their loan applicants, and choosing whether or not the applicant is too risky to give the loan to. If the applicant is not, the comapny then determines the terms of the loan. To acquire such applicants, the company uses web/ app sources and lending peer companies, with P2P lending marketplaces. In this project, we are going to assess the quality of all such applicants that a company receives through such marketplaces.

## Business Challenge
The company has tasked you with creating a model that predicts whether or not these leads will complete the **e-sign** phase of the loan application. The company seeks to leverage this model to identify less quality applicants and experiment with giving them different onboarding screens.

## Data
We have access to financial data before the process begins. The data includes personal information like age, and time employed, etc. The comapny utilizes these feature to calculate risk scores based on many different risk factors. In this project, we are given these **risk scores**. Furthermore, the marketplace provides us with their own lead quality scores. We will leverage this data to predict if the user is likely to respond to our current onboarding process.

## Step 1: Importing Libraries and Data
The initial section involves importing essential Python libraries such as Pandas, NumPy, Matplotlib, and Seaborn. The dataset (P39-Financial-Data.csv) is loaded into a Pandas DataFrame (df).

## Step 2: Data Visualization
### Cleaning the Data
Descriptive statistics of the dataset are presented using the describe() method. Additionally, the presence of any missing values is checked using the isna().any() method, indicating that there are no missing values.

### Histograms
Histograms of numerical columns are plotted to visualize the distribution of data, providing insights into the characteristics of different features.

![image](https://github.com/Devansh-Gupta-Official/likelihood-e-signing-loan/assets/100591612/6bc58d5b-5577-46d8-8190-cae9dc2e7cea)


### Correlation Analysis
- **Correlation Plot (with response variable)**
A bar chart is created to display the correlation of each feature with the response variable (e_signed). This helps understand the linear relationship between features and the target variable.

![image](https://github.com/Devansh-Gupta-Official/likelihood-e-signing-loan/assets/100591612/d9a59fce-5148-4fbc-a42e-30a71121e3e7)


- **Correlation Matrix**
A heatmap visualizes the correlation matrix, allowing for a comprehensive view of the pairwise relationships between features.

![image](https://github.com/Devansh-Gupta-Official/likelihood-e-signing-loan/assets/100591612/77d76f57-b3cd-4869-8ec5-a41ad989a80d)


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
  - Logistic Regression is used as a baseline model. The LogisticRegression class from scikit-learn is employed, with L1 regularization (Lasso) to potentially select important features. The model is trained on the training set and evaluated on the test set using metrics such as accuracy, precision, recall, and F1 score.

- **Support Vector Machines (SVM)**

- **1. SVM with Linear Kernel**
A Support Vector Machine (SVM) with a linear kernel is implemented using the SVC class from scikit-learn. The model is trained and evaluated similarly to the Logistic Regression model, and performance metrics are recorded.

- **2. SVM with Radial Basis Function (RBF) Kernel**
Another SVM model is created, this time using an RBF kernel. The same training and evaluation process is followed, and the model's performance is compared with the linear SVM.

- **Random Forest Model**
  - A Random Forest classifier is employed for more complex, ensemble-based modeling. The RandomForestClassifier class from scikit-learn is used with 100 decision trees and entropy as the criterion. The model is trained on the training set and evaluated on the test set, recording performance metrics.

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


## Results
### Baseline Models
**1. Logistic Regression (Lasso)**
 - Accuracy: 56.20%
 - Precision: 57.60%
 - Recall: 70.59%
 - F1 Score: 63.43%

**2. SVM (Linear)**
 - Accuracy: 56.84%
 - Precision: 57.78%
 - Recall: 73.55%
 - F1 Score: 64.72%

**3. SVM (RBF)**
 - Accuracy: 59.16%
 - Precision: 60.57%
 - Recall: 69.09%
 - F1 Score: 64.55%

**4. Random Forest**
 - Accuracy: 62.17%
 - Precision: 64.01%
 - Recall: 67.89%
 - F1 Score: 65.89%

### Model Improvement
**K-Fold Cross Validation**

The average accuracy across 10 folds: 63.00%
This helps assess the generalization performance of the models and identify potential overfitting or underfitting.


**1. Grid Search for Random Forest**
- Round 1: Entropy
 - Best Accuracy: 63.45%
 - Optimal Hyperparameters:
  - Bootstrap: True
  - Criterion: Entropy
  - Max Depth: None
  - Max Features: 5
  - Min Samples Leaf: 5
  - Min Samples Split: 2
- Round 2: Gini
 - Best Accuracy: 63.46%
 - Optimal Hyperparameters:
  - Bootstrap: False
  - Criterion: Gini
  - Max Depth: None
  - Max Features: 6
  - Min Samples Leaf: 1
  - Min Samples Split: 10

**2. Testing New Parameters on Test Set**
- Random Forest (Grid Search: Entropy)
 - Accuracy: 63.07%
 - Precision: 64.52%
 - Recall: 69.71%
 - F1 Score: 67.02%

- Random Forest (Grid Search: Gini)
 - Accuracy: 62.81%
 - Precision: 64.54%
 - Recall: 68.62%
 - F1 Score: 66.52%


