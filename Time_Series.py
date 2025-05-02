import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

df = pd.read_excel("Menomonie_Weather.xlsx")
df.dropna(inplace=True)

plt.figure(figsize=(14, 7))
plt.plot(df.index, df["Avg_Temp"], label='Average Temp')
plt.xlabel('Date')
plt.ylabel('Average Temp')
plt.legend()
plt.show()

original = adfuller(df["Avg_Temp"])

print(f"ADF Statistic (Original): {original[0]:.4f}")
print(f"p-value (Original): {original[1]:.4f}")

if original[1] < 0.05:
    print("Interpretation: The original series is Stationary.\n")
else:
    print("Interpretation: The original series is Non-Stationary.\n")

# Apply first-order differencing
df['Avg_Temp_Diff'] = df['Avg_Temp'].diff()

# Perform the Augmented Dickey-Fuller test on the differenced series
result_diff = adfuller(df["Avg_Temp_Diff"].dropna())
print(f"ADF Statistic (Differenced): {result_diff[0]:.4f}")
print(f"p-value (Differenced): {result_diff[1]:.4f}")
if result_diff[1] < 0.05:
    print("Interpretation: The differenced series is Stationary.")
else:
    print("Interpretation: The differenced series is Non-Stationary.")

plt.figure(figsize=(14, 7))
plt.plot(df.index, df['Avg_Temp_Diff'], label='Differenced Avg_Temp Price', color='orange')
plt.xlabel('Date')
plt.ylabel('Differenced Avg_Temp Price')
plt.legend()
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(16, 4))

# ACF plot
plot_acf(df['Avg_Temp_Diff'].dropna(), lags=40, ax=axes[0])
axes[0].set_title('Autocorrelation Function (ACF)')

# PACF plot
plot_pacf(df['Avg_Temp_Diff'].dropna(), lags=40, ax=axes[1])
axes[1].set_title('Partial Autocorrelation Function (PACF)')

plt.tight_layout()
plt.show()

# Split data into train and test
train_size = int(len(df) * 0.8)
train, test = df.iloc[:train_size], df.iloc[train_size:]

model = ARIMA(train["Avg_Temp"], order=(1,1,1))
model_fit = model.fit()

forecast = model_fit.forecast(steps=len(test))

plt.figure(figsize=(14,7))
plt.plot(train.index, train["Avg_Temp"], label='Train', color='blue')
plt.plot(test.index, test["Avg_Temp"], label='Test', color='green')
plt.plot(test.index, forecast, label='Forecast', color='orange')
plt.title('Avg_Temp Forecast')
plt.xlabel('Date')
plt.ylabel('Avg_Temp')
plt.legend()
plt.show()

print(f"AIC: {model_fit.aic}")
print(f"BIC: {model_fit.bic}")

forecast = forecast[:len(test)]
test_close = test["Avg_Temp"][:len(forecast)]

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(test_close, forecast))
print(f"RMSE: {rmse:.4f}")