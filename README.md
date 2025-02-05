# Titanic Survival Prediction

## Overview
This project builds a machine learning model to predict the survival of passengers on the Titanic. It uses Logistic Regression and Random Forest classifiers for prediction.

## Dataset
The dataset used is the Titanic dataset, loaded using Seaborn's built-in `sns.load_dataset('titanic')` function.

## Features Used
The following features are selected for training the model:
- `pclass`: Passenger class (1st, 2nd, or 3rd class)
- `sex`: Gender (converted to numerical values: male = 0, female = 1)
- `age`: Passenger age (missing values filled with mean)
- `fare`: Ticket fare
- `sibsp`: Number of siblings/spouses aboard
- `parch`: Number of parents/children aboard
- `embarked`: Port of embarkation (converted to numerical values)
- `family_size`: Total number of family members (calculated as `sibsp + parch + 1`)

## Model Training
Two machine learning models are trained and compared:
1. **Logistic Regression**
2. **Random Forest Classifier**

The dataset is split into training and testing sets (80% training, 20% testing).

## Evaluation Metrics
The models are evaluated using:
- **Accuracy Score**
- **Classification Report** (Precision, Recall, F1-Score)

## Installation and Requirements
To run this project, install the required dependencies using:
```
pip install -r requirements.txt
```

## Running the Code
Execute the script using:
```
python titanic_survival.py
```

## Results
The script prints the accuracy and classification report for both Logistic Regression and Random Forest models to compare their performance.

