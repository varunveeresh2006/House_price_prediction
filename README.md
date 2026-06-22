# House Price Prediction using Machine Learning

## Overview

This project implements a **House Price Prediction** system using **Machine Learning** to estimate residential property prices based on key housing features. The model is trained using historical data and applies **Linear Regression** to predict house prices from attributes such as area, number of bedrooms, bathrooms, house age, and parking availability.

The objective of this project is to demonstrate the complete machine learning workflow, including data preprocessing, exploratory data analysis, model training, prediction, and performance evaluation.

## Features

* Data loading and preprocessing
* Exploratory Data Analysis (EDA)
* Feature selection for model training
* Train-test data splitting
* Linear Regression model implementation
* House price prediction
* Model evaluation using standard regression metrics
* Clean and modular project structure

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-learn
* Jupyter Notebook

## Project Structure

```text
house-price-prediction/
│── data/
│   └── housing.csv
│── notebook.ipynb
│── requirements.txt
│── README.md
│── train.py
```

## Installation

1. Clone or download the project.
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the training script:

```bash
python train.py
```

Or open `notebook.ipynb` in Jupyter Notebook or VS Code to explore the data analysis and model development process interactively.

## Machine Learning Workflow

1. Load the dataset.
2. Perform data inspection and preprocessing.
3. Split the dataset into training and testing sets.
4. Train a Linear Regression model.
5. Generate predictions on unseen data.
6. Evaluate model performance using:

   * Mean Absolute Error (MAE)
   * Mean Squared Error (MSE)
   * R² Score

## Sample Input Features

* Area (Square Feet)
* Number of Bedrooms
* Number of Bathrooms
* House Age
* Parking Spaces

## Expected Output

The trained model predicts the estimated selling price of a house based on the provided input features.

## Future Enhancements

* Support additional regression algorithms such as Random Forest and XGBoost
* Hyperparameter tuning for improved performance
* Interactive web application using Streamlit or Flask
* Model persistence using Joblib
* Deployment to a cloud platform

## Learning Outcomes

This project demonstrates practical experience with:

* Data preprocessing
* Exploratory Data Analysis (EDA)
* Regression modeling
* Machine learning workflows
* Model evaluation
* Python-based data science libraries

## License

This project is intended for educational and portfolio purposes.
