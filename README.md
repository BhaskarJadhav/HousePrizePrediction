# HousePrizePrediction
ML model that predicts the House prize  # California House Price Prediction using XGBoost

## Project Overview
This project aims to predict California house prices using the XGBoost Regressor model. The process involves loading the California Housing dataset, performing exploratory data analysis, training an XGBoost model, evaluating its performance, and finally, demonstrating how to make predictions with new user-provided data.

## Dataset
The dataset used is the California Housing dataset, available through `sklearn.datasets.fetch_california_housing()`. It contains various features related to housing districts in California and their median house values.

## Key Features of the Dataset
The dataset includes the following features:
*   **MedInc**: Median income in block group
*   **HouseAge**: Median house age in block group
*   **AveRooms**: Average number of rooms per household
*   **AveBedrms**: Average number of bedrooms per household
*   **Population**: Block group population
*   **AveOccup**: Average number of household members
*   **Latitude**: Block group latitude
*   **Longitude**: Block group longitude
*   **Price (Target)**: Median house value for California districts

## Dependencies
To run this notebook, you will need the following Python libraries:
*   `numpy`
*   `pandas`
*   `matplotlib`
*   `seaborn`
*   `scikit-learn`
*   `xgboost`

## Project Steps

### 1. Data Loading and Initial Exploration
*   The California Housing dataset is loaded using `sklearn.datasets.fetch_california_housing()`.
*   The data is converted into a Pandas DataFrame, and column names are assigned for clarity.
*   Initial checks for missing values (`isnull().sum()`) and descriptive statistics (`describe()`) are performed.
*   The target variable, 'price', is added to the DataFrame.

### 2. Correlation Analysis
*   A correlation matrix is calculated to understand the relationships between different features and the target variable.
*   A heatmap using `seaborn.heatmap` is generated to visualize these correlations, providing insights into which features are strongly correlated with house prices.

### 3. Data Preprocessing
*   The dataset is split into features (X) and the target variable (y).
*   The data is further divided into training and testing sets using `train_test_split` with a test size of 20% and `random_state=20` for reproducibility.

### 4. Model Training
*   An XGBoost Regressor model (`XGBRegressor()`) is initialized.
*   The model is trained on the `x_train` and `y_train` datasets.

### 5. Model Evaluation
*   **Training Data Evaluation**: Predictions are made on the training data, and the R-squared error and Mean Absolute Error are calculated to assess how well the model learned from the training data.
*   **Test Data Evaluation**: Predictions are made on the unseen test data, and the R-squared error and Mean Absolute Error are calculated to evaluate the model's generalization performance.

### 6. Visualization of Predictions
*   A scatter plot is generated to visualize the relationship between the actual house prices and the predicted house prices from the training set. This helps to visually assess the model's accuracy.

### 7. Making Predictions with User Input
*   The notebook includes a section where a user can input specific house details for the defined features.
*   The trained XGBoost model then uses this input to predict the house price.

## How to Run
1.  Ensure you have all the dependencies installed (`pip install numpy pandas matplotlib seaborn scikit-learn xgboost`).
2.  Open the Jupyter/Colab notebook.
3.  Run all cells sequentially to execute the data loading, preprocessing, model training, and evaluation steps.
4.  Interact with the final code cell to input your own house details and get a price prediction.
