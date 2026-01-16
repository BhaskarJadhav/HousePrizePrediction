#  Housing Price Prediction

## Project Overview
This project aims to build a machine learning model to accurately predict median house prices in California using the California Housing dataset. The goal is to provide a predictive tool that can estimate house values based on various socio-economic and geographical factors.

## Dataset
The dataset used for this project is the California Housing dataset, which is publicly available through scikit-learn's `sklearn.datasets.fetch_california_housing()` function.

### Data Characteristics:
*   **Features:** The dataset includes 8 features:
    *   `MedInc`: Median income in block group
    *   `HouseAge`: Median house age in block group
    *   `AveRooms`: Average number of rooms per household
    *   `AveBedrms`: Average number of bedrooms per household
    *   `Population`: Block group population
    *   `AveOccup`: Average number of household members
    *   `Latitude`: Block group latitude
    *   `Longitude`: Block group longitude
*   **Target Variable:** `price` (Median house value for California districts, expressed in hundreds of thousands of dollars ($100,000)).
*   **Size:** The dataset contains 20,640 samples.

### Data Preprocessing:
*   The raw data was loaded into a Pandas DataFrame.
*   A new column named 'price' was added to the DataFrame to store the target variable.
*   Missing values were checked across all features and the target; no missing values were found.
*   Descriptive statistics were generated to understand the distribution and central tendency of the data.
*   Correlation between features and the target was analyzed using `df.corr()` and visualized with a heatmap to identify relationships.

## Model
*   **Model Type:** XGBoost Regressor.
*   **Reasoning:** XGBoost (eXtreme Gradient Boosting) is chosen for its strong performance in various machine learning tasks, especially for structured data. It is an ensemble learning method that builds a strong predictive model from a combination of weak prediction models (decision trees).

## Training
*   **Data Splitting:** The dataset was split into training and testing sets with an 80:20 ratio, respectively, using `train_test_split` with a `random_state` of 20 for reproducibility.
    *   `x_train`: Training features.
    *   `x_test`: Testing features.
    *   `y_train`: Training target (actual prices).
    *   `y_test`: Testing target (actual prices).
*   **Model Training:** The XGBoost Regressor model was trained using the `x_train` and `y_train` datasets.

## Evaluation
The model's performance was evaluated using two common regression metrics:
*   **R-squared (R2) Score:** Measures how well the predictions approximate the real data points. An R2 score of 1 indicates that the model perfectly predicts the target values.
*   **Mean Absolute Error (MAE):** Measures the average magnitude of the errors in a set of predictions, without considering their direction. A lower MAE indicates a better model.

### Evaluation Results:
*   **On Training Data:**
    *   R-squared Error: Approximately 0.97 (97% of the variance in the dependent variable is predictable from the independent variables).
    *   Mean Absolute Error: Approximately 0.19.
*   **On Test Data:**
    *   R-squared Error: Approximately 0.84 (The model generalizes well to unseen data).
    *   Mean Absolute Error: Approximately 0.31.

### Visualization:
*   A scatter plot was generated to visualize the relationship between the actual prices and the predicted prices from the training set, confirming a strong linear correlation between actual and predicted values.

## Usage
To use the trained model for predicting house prices, you need to provide the following features for a house:
*   `MedInc`
*   `HouseAge`
*   `AveRooms`
*   `AveBedrms`
*   `Population`
*   `AveOccup`
*   `Latitude`
*   `Longitude`

The model will then output the predicted house price in hundreds of thousands of dollars.
