import pandas as pd
import requests
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller

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
def load_and_preprocess_data(data):
    df = pd.DataFrame(data)
    df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Description'], format='%d-%m-%Y %H:%M')
    df = df.sort_values(by='DateTime')
    df.set_index('DateTime', inplace=True)

    df['kVAh'] = pd.to_numeric(df['kVAh'], errors='coerce')
    df['kVAh'].fillna(0, inplace=True)

    return df

def filter_data_by_date(processed_data, start_date, end_date):
    print("Processed Data:", processed_data)  # Debugging line
    print("Start Date:", start_date, "End Date:", end_date)  # Debugging line
    
    # Convert start and end dates to datetime objects
    start_date_dt = pd.to_datetime(start_date, format='%Y-%m-%d')
    end_date_dt = pd.to_datetime(end_date, format='%Y-%m-%d')

    # Use the index for filtering since processed_data should have DateTime as index
    mask = (processed_data.index >= start_date_dt) & (processed_data.index <= end_date_dt)
    filtered_data = processed_data.loc[mask]

    if filtered_data.empty:
        return None  # or raise an error indicating no data found

    return filtered_data

def check_stationarity(timeseries, adf_threshold=0.05):
    result = adfuller(timeseries)
    return result[1] < adf_threshold

def prepare_hourly_data(filtered_data):
    filtered_data['kVah_diff'] = filtered_data['kVAh'].diff().abs()
    hourly_kvah = filtered_data['kVah_diff'].dropna().resample('h').sum()
    
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
