import requests
from flask import Flask, request, jsonify
from forecasting import load_and_preprocess_data, filter_data_by_date, prepare_hourly_data, sarima_forecast
from billing import calculate_energy_bill, get_billing_rates
from datetime import datetime
import os

app = Flask(__name__)

# Dictionary to store forecast data for retrieval
forecast_data_storage = {
    "actual_hourly_kVAh": [],
    "forecasted_kVAh": []
}

@app.route('/')
def home():
    """Return basic API information."""
    return jsonify({
        "message": "Welcome to the Energy Forecasting API",
        "endpoints": {
            "/forecast": "POST: Submit start_date, end_date, and forecast_hours to get forecast and billing info",
            "/api/forecast-data": "GET: Retrieve the most recent forecast data"
        }
    })

@app.route('/forecast', methods=['POST'])
def forecast():
    """Generate a forecast based on user input, return billing details as JSON."""
    api_url = "https://render-ivuy.onrender.com/data"  # The input data API

    # Get user input from the POST request body
    start_date = request.json.get('start_date')
    end_date = request.json.get('end_date')
    forecast_hours = request.json.get('forecast_hours', 24)  # Default to 24 hours if not provided

    # Validate input
    try:
        # Validate date formats
        start_date_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_date_dt = datetime.strptime(end_date, '%Y-%m-%d')

        if start_date_dt >= end_date_dt:
            return jsonify({"error": "Start date must be before end date."}), 400

        # Validate forecast_hours
        forecast_hours = int(forecast_hours)
        if forecast_hours <= 0:
            return jsonify({"error": "Forecast hours must be a positive integer."}), 400

    except ValueError as e:
        return jsonify({"error": f"Invalid input: {str(e)}"}), 400

    # Fetch the data from the external API
    try:
        response = requests.get(api_url)
        response.raise_for_status()  # Check for request errors
        data = response.json()  # Load the data from the API response
    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Failed to fetch data from API: {str(e)}"}), 500

    # Preprocess the loaded data
    try:
        processed_data = load_and_preprocess_data(data)
    except Exception as e:
        return jsonify({"error": f"Data preprocessing failed: {str(e)}"}), 500

    # Filter the data by the requested date range
    try:
        filtered_data = filter_data_by_date(processed_data, start_date, end_date)
        if filtered_data is None:
            return jsonify({"error": "No data available for the given date range."}), 404
    except Exception as e:
        print(f"Date filtering failed: {str(e)}")  # Debugging line
        return jsonify({"error": f"Date filtering failed: {str(e)}"}), 400

    # Prepare the hourly data for forecasting
    try:
        hourly_kvah_diff, hourly_kvah = prepare_hourly_data(filtered_data)
    except Exception as e:
        return jsonify({"error": f"Error preparing hourly data: {str(e)}"}), 500

    # Perform SARIMA forecasting
    try:
        forecast_df, _ = sarima_forecast(hourly_kvah_diff, (1, 0, 1), (1, 1, 1, 24), forecast_hours)
    except Exception as e:
        return jsonify({"error": f"Forecasting failed: {str(e)}"}), 500

    # Calculate billing based on forecasted data
    rates = get_billing_rates()
    total_hours = len(forecast_df)
    charges = calculate_energy_bill(forecast_df, rates, total_hours)

    # Prepare actual and forecasted data for storage
    actual_hourly_kvah = hourly_kvah.reset_index()
    actual_hourly_kvah['DateTime'] = actual_hourly_kvah['DateTime'].dt.strftime('%Y-%m-%d %H:%M:%S')

    forecasted_kvah = forecast_df[['Date_Hourly', 'Forecasted_kVah']].copy()
    forecasted_kvah['DateTime'] = forecasted_kvah['Date_Hourly'].dt.strftime('%Y-%m-%d %H:%M:%S')
    forecasted_kvah = forecasted_kvah.drop(columns=['Date_Hourly']).to_dict(orient='records')

    # Store the actual and forecasted data for later retrieval
    forecast_data_storage["actual_hourly_kVAh"] = actual_hourly_kvah.to_dict(orient='records')
    forecast_data_storage["forecasted_kVAh"] = forecasted_kvah

    # Return the billing information with detailed charges as JSON
    return jsonify({
        'billing_info': {
            'total_charges': charges['total_charges'],
            'demand_charges': charges['demand_charges'],
            'wheeling_charges': charges['wheeling_charges'],
            'energy_charges': charges['energy_charges'],
            'tod_charges': charges['tod_charges'],
            'fac': charges['fac'],
            'electricity_duty': charges['electricity_duty'],
            'tax_on_sale': charges['tax_on_sale'],
        }
    })

@app.route('/api/forecast-data', methods=['GET'])
def get_forecast_data():
    """Retrieve the most recent forecast data."""
    if not forecast_data_storage["actual_hourly_kVAh"] and not forecast_data_storage["forecasted_kVAh"]:
        return jsonify({"message": "No forecast data available"}), 404

    return jsonify({
        'actual_hourly_kVAh': forecast_data_storage["actual_hourly_kVAh"],
        'forecasted_kVAh': forecast_data_storage["forecasted_kVAh"]
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
