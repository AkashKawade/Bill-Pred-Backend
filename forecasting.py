import pandas as pd
import requests
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt

# Fetch data from API
def fetch_data_from_api(api_url, params=None):
    try:
        response = requests.get(api_url, params=params)
        if response.status_code == 200:
            data = response.json()
            return data
        else:
            print(f"Failed to fetch data. Status code: {response.status_code}")
            return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# Load and preprocess data from API
def load_and_preprocess_data(api_url):
    data = fetch_data_from_api(api_url)
    if data is None:
        return None

    df = pd.DataFrame(data)
    df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Description'], format='%d-%m-%Y %H:%M')
    df = df.sort_values(by='DateTime')
    df.set_index('DateTime', inplace=True)

    df['kVAh'] = pd.to_numeric(df['kVAh'], errors='coerce')
    df['kVAh'].fillna(0, inplace=True)

    return df

def filter_data_by_date(df, start_date, end_date):
    mask = (df.index >= start_date) & (df.index <= end_date)
    return df[mask].copy()

def check_stationarity(timeseries, adf_threshold=0.05):
    result = adfuller(timeseries)
    return result[1] < adf_threshold

def prepare_hourly_data(filtered_data):
    filtered_data['kVah_diff'] = filtered_data['kVAh'].diff().abs()
    hourly_kvah = filtered_data['kVah_diff'].dropna().resample('H').sum()
    
    if not check_stationarity(hourly_kvah):
        hourly_kvah_diff = hourly_kvah.diff().dropna()
    else:
        hourly_kvah_diff = hourly_kvah
    
    return hourly_kvah_diff, hourly_kvah

def sarima_forecast(time_series, order, seasonal_order, n_hours):
    model = SARIMAX(time_series, order=order, seasonal_order=seasonal_order)
    results = model.fit(disp=False)
    forecast = results.get_forecast(steps=n_hours)
    conf_int = forecast.conf_int()

    future_dates = pd.date_range(time_series.index[-1] + pd.Timedelta(hours=1), periods=n_hours, freq='H')

    forecast_df = pd.DataFrame({
        'Date_Hourly': future_dates,
        'Forecasted_kVah': forecast.predicted_mean,
        'Lower_CI_kVah': conf_int.iloc[:, 0],
        'Upper_CI_kVah': conf_int.iloc[:, 1]
    })

    return forecast_df, results

def plot_forecast(original_data, forecast_df, title):
    plt.figure(figsize=(12, 6))
    plt.plot(original_data.index, original_data, label='Observed Hourly kVah', alpha=0.5)
    plt.plot(forecast_df['Date_Hourly'], forecast_df['Forecasted_kVah'], color='red', label='Forecasted Hourly kVah')
    plt.fill_between(forecast_df['Date_Hourly'], 
                     forecast_df['Lower_CI_kVah'], 
                     forecast_df['Upper_CI_kVah'], 
                     color='pink', alpha=0.5, label='Confidence Interval')
    plt.title(title)
    plt.xlabel('Date & Time')
    plt.ylabel('kVah')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    
