# Fish Market Polynomial Regression: Algorithm & Dataset Explanation

## 1️⃣ Introduction
The **Fish Market Dataset** is a collection of data that contains various physical characteristics of different fish species. The objective of this project is to **predict the weight of a fish** based on its physical attributes using **Polynomial Regression**. This document provides a detailed explanation of the dataset and the algorithm used.

---

## 2️⃣ Dataset Description
The dataset consists of **159 rows** and **7 columns**, where each row represents an individual fish. The dataset is structured as follows:

### **Features & Target Variable**
| Column Name  | Description |
|-------------|-------------|
| **Species** | Categorical variable representing the fish species (e.g., Perch, Bream, Roach, Pike, Smelt, Parkki, Whitefish). |
| **Weight**  | Target variable: The weight of the fish in grams. |
| **Length1** | First length measurement of the fish (cm). |
| **Length2** | Second length measurement of the fish (cm). |
| **Length3** | Third length measurement of the fish (cm). |
| **Height**  | The height of the fish (cm). |
| **Width**   | The width of the fish (cm). |

### **Understanding the Dataset**
- The **Species** column is a **categorical variable**, which needs to be converted into a numerical format using **one-hot encoding**.
- **Weight** is the **dependent variable (target)** we want to predict.
- The **other numerical features** serve as the **independent variables** for prediction.
- There are **no missing values** in the dataset, ensuring data completeness.

---

## 3️⃣ Algorithm: Polynomial Regression
Polynomial Regression is an extension of Linear Regression that **models non-linear relationships** between independent and dependent variables by introducing polynomial terms.

### **3.1 How Polynomial Regression Works**
- **Step 1: Transform Features** – The original features are expanded into polynomial terms, such as:
  - For a single feature \(x\), a **degree-2** polynomial regression would create new terms: \(x^2\), and for **degree-3**: \(x^3\), etc.
  - If we have multiple features \((x_1, x_2)\), polynomial terms include \(x_1^2, x_2^2, x_1x_2\), etc.

- **Step 2: Apply Standard Linear Regression** – The new polynomial features are used in a standard linear regression model to find the best-fitting coefficients.

- **Step 3: Predict and Evaluate** – The model predicts fish weight based on the transformed features and evaluates performance using metrics like **R² score**.

### **3.2 Mathematical Representation**
For a polynomial regression of degree **2**, the equation takes the form:

\[ y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \beta_3 x_1^2 + \beta_4 x_2^2 + \beta_5 x_1 x_2 + \epsilon \]

Where:
- \( y \) is the predicted **fish weight**.
- \( x_1, x_2 \) are the independent features (e.g., Height, Width).
- \( \beta_0, \beta_1, \beta_2, ... \) are the regression coefficients.
- \( \epsilon \) is the error term.

For **higher degrees (e.g., 3 or 4),** more polynomial terms are added to capture complex relationships.

---

## 4️⃣ Implementation Workflow
### **4.1 Data Preprocessing**
1. **Rename columns** for better readability.
2. **Remove anomalies** (e.g., erroneous weight values).
3. **One-hot encode categorical variables** (e.g., Species → binary columns).

### **4.2 Model Training**
1. **Split data into Training & Test sets** (80%-20%).
2. **Apply Polynomial Feature Transformation** (`degree=2`).
3. **Scale Features** using `StandardScaler()`.
4. **Train a Linear Regression model** on transformed data.

### **4.3 Model Evaluation**
1. **R² Score:** Measures how well the model explains variance.
2. **Scatter Plot:** Compares actual vs predicted weights.
3. **Residual Analysis:** Checks error distribution.

---

## 5️⃣ Results & Findings
- **Polynomial Regression (Degree 2) achieved an R² score of 0.9282**, indicating a **strong predictive performance**.
- **Visual analysis confirmed that polynomial regression effectively captures non-linear patterns** in fish weight prediction.

### **Potential Improvements**
✅ Increase polynomial degree to 3 for complex relationships.
✅ Apply feature selection to improve computational efficiency.
✅ Implement cross-validation for better generalization.

---

## 6️⃣ Conclusion
Polynomial Regression effectively models the relationship between a fish’s physical characteristics and its weight. The high **R² score (92.82%)** demonstrates that **height, width, and species information significantly contribute to accurate weight prediction**.


