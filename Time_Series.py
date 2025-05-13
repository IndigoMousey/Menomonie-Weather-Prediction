import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Load and preprocess data
def load_weather_data(file_path):
    df = pd.read_csv(file_path)
    df.dropna(inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df['Avg_Temp'] = df['Avg_Temp'].astype(float)
    df['Precipitation'] = df['Precipitation'].astype(float)
    return df

# Forecast function
def forecast_variable(series, label, train_days=15, forecast_days=30):

    start_index = - (train_days + forecast_days)
    test_index = -forecast_days
    end_index = None  # till the end

    train = series.iloc[start_index:test_index+1]
    test = series.iloc[test_index:end_index]

    # print(f"\n--- Auto ARIMA for {label} ---")
    # auto_model = auto_arima(
    #     train,
    #     seasonal=True,
    #     stepwise=False,
    #     suppress_warnings=True,
    #     trace=True,
    # )
    # best_order = auto_model.order
    # print(f"Selected ARIMA order for {label}: {best_order}")

    model = ARIMA(train, order=(5,1,1) )
    model_fit = model.fit()

    forecast_result = model_fit.get_forecast(steps=forecast_days)
    forecast = forecast_result.predicted_mean

    forecast_index = test.index
    rmse = np.sqrt(mean_squared_error(test, forecast))
    print(f"RMSE for {label} forecast: {rmse:.2f}")

    # Plot
    plt.figure(figsize=(14, 7))
    plt.plot(test.index, test, label='Actual', color='red')
    plt.plot(forecast_index, forecast, label='Forecast', color='orange')
    plt.title(f'{label} Forecast')
    plt.xlabel('Date')
    plt.ylabel(label)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# Main script execution
def main():
    #Load dataset
    file_path = "Menomonie_Weather.csv"
    df = load_weather_data(file_path)

    #Create time series for Temperature
    forecast_variable(df['Max_Temp'], "Max_Temp")
    # forecast_variable(df['Min_Temp'], "Min_Temp")
    # forecast_variable(df['Avg_Temp'], "Avg_Temp")


if __name__ == "__main__":
    main()
