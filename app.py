from flask import Flask, render_template, request, jsonify
from forecasting import load_and_preprocess_data, filter_data_by_date, prepare_hourly_data, sarima_forecast, plot_forecast
from billing import calculate_energy_bill, get_billing_rates
import pandas as pd
import io
import base64
import json
import matplotlib.pyplot as plt

app = Flask(__name__)

# Global variable to store the JSON data
json_data = {}

@app.route('/')
def home():
    api_url = "https://render-ivuy.onrender.com/data"
    data = load_and_preprocess_data(api_url)
    
    input_data = data.head(100).reset_index().to_dict(orient='records')  # Prepare data for HTML rendering
    
    return render_template('index.html', input_data=input_data)

@app.route('/forecast', methods=['POST'])
def forecast():
    global json_data  # Access the global variable
    
    api_url = "https://render-ivuy.onrender.com/data"

    # Get user input
    start_date = request.form['start_date']
    end_date = request.form['end_date']
    forecast_hours = int(request.form['forecast_hours'])

    # Load and preprocess data
    data = load_and_preprocess_data(api_url)
    if data is None:
        return jsonify({"error": "Failed to load data from API"}), 500

    filtered_data = filter_data_by_date(data, start_date, end_date)

    # Prepare hourly data for forecasting
    try:
        hourly_kvah_diff, hourly_kvah = prepare_hourly_data(filtered_data)
    except Exception as e:
        return jsonify({"error": f"Error preparing hourly data: {str(e)}"}), 500

    # Forecasting using SARIMA
    try:
        forecast_df, _ = sarima_forecast(hourly_kvah_diff, (1, 0, 1), (1, 1, 1, 24), forecast_hours)
    except Exception as e:
        return jsonify({"error": f"Forecasting failed: {str(e)}"}), 500

    # Billing calculation
    rates = get_billing_rates()
    total_hours = len(forecast_df)
    charges = calculate_energy_bill(forecast_df, rates, total_hours)

    # Prepare actual and forecasted data for JSON response
    actual_hourly_kvah = hourly_kvah.reset_index()
    actual_hourly_kvah['DateTime'] = actual_hourly_kvah['DateTime'].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    forecasted_kvah = forecast_df[['Date_Hourly', 'Forecasted_kVah']].copy()
    forecasted_kvah['DateTime'] = forecasted_kvah['Date_Hourly'].dt.strftime('%Y-%m-%d %H:%M:%S')
    forecasted_kvah = forecasted_kvah.drop(columns=['Date_Hourly']).to_dict(orient='records')

    # Create a structured JSON object
    json_data = {
        'actual_hourly_kVAh': actual_hourly_kvah.to_dict(orient='records'),
        'forecasted_kVAh': forecasted_kvah
    }

    return render_template('results.html', charges=charges)


@app.route('/api/forecast-data', methods=['GET'])
def get_forecast_data():
    return jsonify(json_data)  # Send the JSON data as response

if __name__ == '__main__':
    app.run()
