import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Load dataset
df = pd.read_csv("Fish.csv")  # Ensure this file is in the same directory
print('Shape of dataset:', df.shape)

# Rename columns for clarity
df.rename(columns={'Length1': 'VerticalLen', 'Length2': 'DiagonalLen', 'Length3': 'CrossLen'}, inplace=True)

# Remove invalid weight entry
df = df[df["Weight"] > 0]

# Check for missing values
print("Missing values per column:\n", df.isna().sum())

# Encoding categorical variable 'Species' using one-hot encoding
df = pd.get_dummies(df, columns=['Species'], drop_first=True)

# Select features (X) and target variable (y)
X = df.drop(columns=['Weight'])  # Independent variables (lengths, height, width, species)
y = df['Weight']  # Dependent variable

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)

# Apply Polynomial Feature Transformation (degree=2 for quadratic relationships)
degree = 2  # Change degree to experiment with model complexity
poly_model = make_pipeline(PolynomialFeatures(degree), StandardScaler(), LinearRegression())

# Train model
poly_model.fit(X_train, y_train)

# Predictions
y_pred = poly_model.predict(X_test)

# Evaluate model performance
r2 = r2_score(y_test, y_pred)
print(f"Polynomial Regression (Degree {degree}) R² Score: {r2:.4f}")

# Visualization - Actual vs Predicted
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)  # Ideal prediction line
plt.xlabel("Actual Weight")
plt.ylabel("Predicted Weight")
plt.title(f"Actual vs Predicted Weight (Polynomial Degree {degree})")
plt.show()

# Residual Distribution
plt.figure(figsize=(10, 6))
sns.histplot(y_test - y_pred, kde=True, bins=20)
plt.title("Residual Distribution")
plt.xlabel("Prediction Error")
plt.show()
