# Heart Disease Prediction

## Description
This project applies machine learning techniques to predict the likelihood of heart disease in patients based on various health attributes such as age, blood pressure, cholesterol, and more. The goal is to build a classification model using popular machine learning algorithms to assess the risk of heart disease.

## Technologies Used
- **Python**: For implementing machine learning models.
- **Scikit-learn**: For building and evaluating classification models.
- **Pandas**: For data manipulation and preprocessing.
- **Matplotlib/Seaborn**: For data visualization and analysis.
- **Jupyter Notebook**: For interactive data analysis and model training.

## Installation Instructions
Follow these steps to set up the project locally.

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/samia2111/heart-disease-prediction.git
    ```
2. **Install Dependencies**:
    If you don’t have `pip` installed, use the following command:
    ```bash
    pip install -r requirements.txt
    ```
3. **Run the Project**:
    After installation, you can run the project with the following:
    ```bash
    python heart_disease_prediction.py
    ```

## Usage
### Example Code:
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv('heart_disease_data.csv')

# Preprocessing data (if necessary, handle missing values, normalization)
# Example: data.fillna(0, inplace=True)

# Train-test split
X = data.drop('target', axis=1)  # Features
y = data['target']  # Target variable (Heart Disease: 1 = Yes, 0 = No)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Example predictions
print("Predictions:", y_pred)
