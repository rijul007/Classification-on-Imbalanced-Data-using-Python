# House Price Prediction using Machine Learning

## Objective
The objective of this project is to build a machine learning model that predicts house prices based on various features such as lot size, building type, year built, and more. This project aims to assist potential buyers and sellers in making informed decisions by providing accurate price predictions.


## Dataset Used
The dataset used for this project contains 13 features and 2919 records. It includes the following key attributes:
1. **Id**: Record identifier.
2. **MSSubClass**: Type of dwelling involved in the sale.
3. **MSZoning**: General zoning classification of the sale.
4. **LotArea**: Lot size in square feet.
5. **LotConfig**: Configuration of the lot.
6. **BldgType**: Type of dwelling.
7. **OverallCond**: Overall condition of the house.
8. **YearBuilt**: Original construction year.
9. **YearRemodAdd**: Remodel date.
10. **Exterior1st**: Exterior covering on the house.
11. **BsmtFinSF2**: Type 2 finished square feet.
12. **TotalBsmtSF**: Total square feet of basement area.
13. **SalePrice**: Target variable to be predicted.


## Analysis Technique

### **1. Data Preprocessing**
- **Categorization of Features**: Features were categorized based on their data types (categorical, integer, float).
  - Categorical variables: 4
  - Integer variables: 6
  - Float variables: 3
- **Handling Missing Values**:
  - Columns irrelevant to prediction (e.g., `Id`) were dropped.
  - Missing values in `SalePrice` were replaced with the mean value to ensure a symmetric data distribution.
  - Records with null values in other features were dropped if their count was minimal.
- **OneHotEncoding**: Categorical features were converted into binary vectors using OneHotEncoder to make them suitable for machine learning models.

---

### **2. Exploratory Data Analysis (EDA)**
EDA was conducted to uncover patterns and relationships in the data:
- **Heatmap**: A correlation heatmap was created using the Seaborn library to identify relationships between features and the target variable (`SalePrice`).
- **Barplots**: Barplots were used to analyze the distribution of categorical features like `Exterior1st`, which has 16 unique categories. This helped in understanding the frequency of each category.

---

### **3. Model Selection and Training**
The following regression models were used to predict house prices:
- **Support Vector Machine (SVM)**:
  - SVM was used for regression by finding the optimal hyperplane in an n-dimensional space.
  - **Mean Absolute Percentage Error (MAPE)**: 0.18705129
- **Random Forest Regressor**:
  - An ensemble technique that uses multiple decision trees for regression.
  - **MAPE**: 0.1929469
- **Linear Regression**:
  - A simple regression model that predicts the dependent variable (`SalePrice`) based on independent features.
  - **MAPE**: 0.187416838
- **CatBoost Regressor**:
  - A gradient boosting algorithm optimized for categorical data.
  - **MAPE**: 0.383511698


## Result
The **Support Vector Machine (SVM)** model achieved the best performance with the lowest Mean Absolute Percentage Error (MAPE) of **0.187**. This indicates that the SVM model is the most accurate among the models tested for predicting house prices. However, further improvements can be made using ensemble techniques like Bagging and Boosting.

For a detailed view of the analysis and visualizations, you can access the [Jupyter Notebook here](https://nbviewer.org/urls/gist.githubusercontent.com/rijul007/a6eb2332e6451873701971ac4ca124ab/raw/62303d08e48f08bc3aff4f332f0642dc746d70f2/hpp-notebook.ipynb).