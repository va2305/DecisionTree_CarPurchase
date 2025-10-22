# Decision Tree Car Purchase Prediction

## Overview
This project uses a **Decision Tree Classifier** to predict whether a person will purchase a car based on features like **Gender, Age, and Salary**. The dataset contains customer information and their purchasing decision.

## Dataset
- File: `car_data.csv`
- Features:
  - Gender: Male/Female
  - Age: Customer age
  - Salary: Annual salary
  - Purchased: 0 (No) / 1 (Yes)

## Libraries Used
- pandas
- scikit-learn
- matplotlib

## Steps
1. Load the dataset.
2. Encode categorical data (Gender).
3. Split data into training and testing sets.
4. Train a Decision Tree Classifier.
5. Evaluate model performance using accuracy and classification report.
6. Visualize the Decision Tree.
7. Predict for new data points.

## Usage
```bash
# Run the project
python decision_tree_car.py
