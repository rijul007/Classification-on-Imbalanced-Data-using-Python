# Classification on Imbalanced Data using Python

## Objective
This project aims to develop a classification model on an imbalanced dataset. The primary goal is to handle the challenges associated with imbalanced class distribution, particularly in predicting the minority class effectively.

## Dataset Used
The dataset used for this project contains insurance policy records, including customer demographics, vehicle details, and the `claim_status` target variable. The data has 58,592 entries and 41 columns, with a significant imbalance between the classes (claims vs. no claims).

## Analysis Technique
To address the class imbalance, the following techniques were used extensively throughout the analysis:

1. **Data Preprocessing**:
   - **Exploratory Data Analysis (EDA)**: The data was explored to understand the distribution of numerical and categorical features.
   - **Handling Missing Values**: Checked for missing values and confirmed none were present in the dataset.
   - **Data Visualization**: Plots were generated to examine the class imbalance and feature distributions.

2. **Class Imbalance Handling**:
   - **Oversampling**: Synthetic Minority Over-sampling Technique (SMOTE) was used to balance the classes. This was crucial because the dataset showed a heavy imbalance, with significantly fewer claims (`claim_status = 1`) than non-claims (`claim_status = 0`).
   - **Resampling**: By creating synthetic samples of the minority class, we ensured that both classes were equally represented in the training data, preventing the model from becoming biased towards the majority class.

3. **Feature Selection**:
   - Analyzed the importance of both numerical and categorical variables using feature importance metrics. Key features like `policy_id`, `subscription_length`, and `customer_age` were identified as the most influential in predicting the claim status.

4. **Model Building**:
   - **Random Forest Classifier**: Chosen for its robustness in handling imbalanced data. The model was trained on the balanced dataset.
   - **Evaluation Metrics**: Precision, Recall, and F1-Score were prioritized over accuracy, as these metrics provide a better understanding of the model’s performance on imbalanced datasets. Additionally, the Area Under the ROC Curve (AUC-ROC) was used to evaluate the overall performance.

## Result
The model achieved a balanced F1-Score, indicating an effective handling of the minority class. Precision and recall metrics showed a significant improvement after balancing the dataset.