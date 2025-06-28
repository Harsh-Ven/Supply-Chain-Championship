#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
REGRESSION 

"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import statsmodels.api as sm

#1. Load the dataset


data = pd.read_csv('M2_R4.csv')



df = pd.DataFrame(data)
print("Data Preview:")
print(df.head())

# 2. Exploratory Data Analysis (EDA)
plt.figure(figsize=(10, 6))
plt.plot(df['Month'], df['Sales'], marker='o', label='Sales')
plt.title('Monthly Sales Trend')
plt.xlabel('Month')
plt.ylabel('Sales (Units)')
plt.grid(True)
plt.legend()
plt.show()

#sns.pairplot(df[["Sales", "Month"]], diag_kind='kde')
#plt.show()

correlation_matrix = df[["Sales", "Month"]].corr()
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
plt.title('Correlation Heatmap')
plt.show()

# 3. Data Preprocessing

#X = df[['Advertising_Spend', 'Discount_Offered']]
#y = df['Sales']

X = df[["Month"]]
y = df["Sales"]

# Train-test split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Train-test split (first 80% as train, last 20% as test)
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# 4. Train Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# Print model summary using statsmodels
X_train_sm = sm.add_constant(X_train)  # Add constant for statsmodels
ols_model = sm.OLS(y_train, X_train_sm).fit()
print(ols_model.summary())

# get residuals
residuals = ols_model.resid

# 5. Model Evaluation
predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)


print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")



# 6. Forecast Future Demand
future_data = {
    "Month": [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
}

future_df = pd.DataFrame(future_data)

future_predictions = model.predict(future_df)

print("Future Demand Forecast:")
for i, pred in enumerate(future_predictions):
    print(f"Month {i + 1}: Predicted Sales = {pred:.2f} units")

# Visualize Predicted vs Actual
plt.figure(figsize=(10, 6))
plt.plot(df['Month'], df['Sales'], marker='o', label='Actual Sales')
plt.plot(df['Month'][:len(y_train)], model.predict(X_train), linestyle='--', label='Predicted Sales (Train)')
plt.scatter(df['Month'][len(y_train):], predictions, color='red', label='Predicted Sales (Test)')
plt.title('Actual vs Predicted Sales')
plt.xlabel('Month')
plt.ylabel('Sales (Units)')
plt.grid(True)
plt.legend()
plt.show()
