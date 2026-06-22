"""
House Price Prediction using Linear Regression

Author: Your Name
Description:
This script loads housing data, trains a Linear Regression model,
evaluates its performance, and predicts house prices.
"""

# Import required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def load_data(file_path):
    """
    Load the dataset from a CSV file.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pandas.DataFrame: Loaded dataset.
    """
    return pd.read_csv(file_path)


def train_model(X_train, y_train):
    """
    Train a Linear Regression model.

    Args:
        X_train: Training features.
        y_train: Training target values.

    Returns:
        Trained LinearRegression model.
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model and print performance metrics.
    """
    predictions = model.predict(X_test)

    print("\n========== Model Evaluation ==========")
    print(f"Mean Absolute Error (MAE): {mean_absolute_error(y_test, predictions):.2f}")
    print(f"Mean Squared Error (MSE): {mean_squared_error(y_test, predictions):.2f}")
    print(f"R² Score: {r2_score(y_test, predictions):.4f}")

    return predictions


def predict_sample(model):
    """
    Predict the price of a sample house.
    """
    sample_house = pd.DataFrame({
        "area_sqft": [1800],
        "bedrooms": [3],
        "bathrooms": [2],
        "house_age_years": [5],
        "parking_spaces": [1]
    })

    predicted_price = model.predict(sample_house)[0]

    print("\n========== Sample Prediction ==========")
    print(f"Estimated House Price: ${predicted_price:,.2f}")


def main():
    """Main execution function."""

    # Load dataset
    df = load_data("data/housing.csv")

    print("First 5 Rows of Dataset:")
    print(df.head())

    # Separate features and target
    X = df.drop("price_usd", axis=1)
    y = df["price_usd"]

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # Train model
    model = train_model(X_train, y_train)
    print("\n✅ Linear Regression model trained successfully!")

    # Evaluate model
    evaluate_model(model, X_test, y_test)

    # Predict for a sample house
    predict_sample(model)


if __name__ == "__main__":
    main()


#Sample output
#First 5 Rows of Dataset:
  # area_sqft  bedrooms  bathrooms  house_age_years  parking_spaces  price_usd
#0       1200         2          2                5               1     250000
#1       1500         3          2                8               1     320000
#2       1800         3          3                2               2     410000
#3       2200         4          3               10               2     520000
#4        950         2          1               15               0     180000

#✅ Linear Regression model trained successfully!

#========== Model Evaluation ==========
#Mean Absolute Error (MAE): 15515.67
#Mean Squared Error (MSE): 303211943.56
#R² Score: 0.9889

#========== Sample Prediction ==========
#Estimated House Price: $381,759.87

