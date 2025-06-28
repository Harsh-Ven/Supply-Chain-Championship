#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SARIMA 

"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load the dataset
df = pd.read_csv("seasonal_demand.csv")
df.set_index('Month', inplace=True)


# Plot the demand over time
plt.figure(figsize=(10,5))
plt.plot(df['Sales'], label='Actual Sales', marker='o')
plt.title('Retail Sales Over Time')
plt.xlabel('Month')
plt.ylabel('Sales')
plt.legend()
plt.show()


#####################################################################
# Check for stationarity using ADF test (find d)

# Perform the Augmented Dickey-Fuller test
def adf_test(series):
    result = adfuller(series)    ## p-value of the test
    print("ADF Statistic:", result[0])
    print("p-value:", result[1])
    print("Critical Values:", result[4])
    if result[1] <= 0.05:
        print("The time series is stationary.")
    else:
        print("The time series is not stationary.")
        
adf_test(df['Sales'])

# If non-stationary, apply first differencing


demand_diff = df['Sales'].diff().dropna()
adf_test(demand_diff)

#####################################################################
# Plot ACF and PACF to identify parameters

# ACF Plot: Used to determine the value of q (MA order)
# Look for the lag where the ACF cuts off sharply (drops to zero).
# If ACF tails off gradually, it suggests a mixed or higher-order MA model.
plot_acf(demand_diff, lags=len(demand_diff)//2-1, title='ACF Plot')
plt.show()

# PACF Plot: Used to determine the value of p (AR order)
# Look for the lag where the PACF cuts off sharply (drops to zero).
# If PACF tails off gradually, it suggests a mixed or higher-order AR model.

plot_pacf(demand_diff, lags= len(demand_diff)//2-1, title='PACF Plot')
plt.show()



#####################################################################
# Fit a SARIMA model with chosen parameters (assumed p=1, d=1, q=3, seasonal p=1, D=1, Q=1, m=4)

# Based on ACF and PACF plots, choose (p, d, q)
# p: Based on PACF cutoff point
# d: Number of differencing steps to make the series stationary
# q: Based on ACF cutoff point

# P → Look for significant PACF spikes at seasonal lags (multiples of m).
# Q → Look for significant ACF spikes at seasonal lags.
# P (Seasonal AR Order): Count significant PACF spikes at seasonal lags (eg. 12, 24, ...).
# Q (Seasonal MA Order): Count significant ACF spikes at seasonal lags (eg. 12, 24, ...).
# If no significant seasonal spikes: P,Q=0

from statsmodels.tsa.stattools import acf, pacf

# Compute ACF and PACF
acf_vals = acf(df["Sales"], nlags=len(demand_diff)//2-1)
pacf_vals = pacf(df["Sales"], nlags=len(demand_diff)//2-1)

# Identify significant seasonal spikes
seasonal_period = 4  # Change this based on data frequency

P = sum(pacf_vals[seasonal_period::seasonal_period] > 0.1)  # Count significant PACF spikes
Q = sum(acf_vals[seasonal_period::seasonal_period] > 0.1)   # Count significant ACF spikes

print(f"Optimal Seasonal P: {P}, Optimal Seasonal Q: {Q}")



####################

p, d, q = 5, 1, 2  # Example values, adjust based on analysis

D,m= 1,4

# P,Q =1,1 # You may want to try other values for P and/or Q


model = SARIMAX(df['Sales'], 
                order=(p,d,q), 
                seasonal_order=(P,D,Q,m), 
                enforce_stationarity=False, 
                enforce_invertibility=False)
results = model.fit()
print(results.summary())

# Plot fitted values vs. actual values
df['SARIMA_Fitted'] = results.fittedvalues

plt.figure(figsize=(10,5))
plt.plot(df['Sales'], label="Actual Demand")
plt.plot(df['SARIMA_Fitted'], label="SARIMA Fitted", linestyle='dashed')
plt.title("SARIMA Model Fit")
plt.legend()
plt.show()

# Forecast next 12 months
step = 12
forecast = results.get_forecast(steps=step)
forecast_mean = forecast.predicted_mean

# forecast_index  = list(range(len(df)+1,(len(df)+1+step)))



plt.figure(figsize=(10,5))
plt.plot(df['Sales'], label="Historical Demand")
plt.plot(forecast_mean.index, forecast_mean, label="Forecast", linestyle='dashed', color='red')
plt.title("SARIMA Forecast for Next 12 Months")
plt.legend()
plt.show()


# Evaluate model performance
mae = mean_absolute_error(df['Sales'].iloc[1:], df['SARIMA_Fitted'].iloc[1:])
rmse = np.sqrt(mean_squared_error(df['Sales'].iloc[1:], df['SARIMA_Fitted'].iloc[1:]))

print(f"Mean Absolute Error: {mae:.2f}")
print(f"Root Mean Squared Error: {rmse:.2f}")


