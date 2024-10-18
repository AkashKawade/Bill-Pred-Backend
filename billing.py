import json

def calculate_energy_bill(forecast_df, rates, total_hours):
    total_predicted_kvah = forecast_df['Forecasted_kVah'].sum()
    predicted_avg_kva = total_predicted_kvah / total_hours
    demand_kva = predicted_avg_kva * 2

    demand_charges = demand_kva * rates['rate_per_kva']
    wheeling_charges = total_predicted_kvah * rates['wheeling_charge_rate']
    energy_charges = total_predicted_kvah * rates['rate_per_kWh']

    tod_intervals = ['0-6', '22-24', '6-9 & 12-18', '9-12', '18-22']
    forecasted_totals = {interval: 0 for interval in tod_intervals}

    for index, row in forecast_df.iterrows():
        hour = row['Date_Hourly'].hour
        kVah = row['Forecasted_kVah']
        
        if 0 <= hour < 6:
            forecasted_totals['0-6'] += kVah
        elif 22 <= hour < 24:
            forecasted_totals['22-24'] += kVah
        elif 6 <= hour < 9 or 12 <= hour < 18:
            forecasted_totals['6-9 & 12-18'] += kVah
        elif 9 <= hour < 12:
            forecasted_totals['9-12'] += kVah
        elif 18 <= hour < 22:
            forecasted_totals['18-22'] += kVah

    tod_charges = sum(forecasted_totals[interval] * rates[f'tod_charge_{interval}'] for interval in tod_intervals)

    fac = total_predicted_kvah * rates['fac_rate']
    electricity_duty = (demand_charges + wheeling_charges + energy_charges + tod_charges + fac) * rates['electricity_duty_rate']
    tax_on_sale = total_predicted_kvah * rates['tax_on_sale_rate']

    total_charges = demand_charges + wheeling_charges + energy_charges + tod_charges + fac + electricity_duty + tax_on_sale
    
    # Return detailed charges
    return {
        'total_charges': total_charges,
        'demand_charges': demand_charges,
        'wheeling_charges': wheeling_charges,
        'energy_charges': energy_charges,
        'tod_charges': tod_charges,
        'fac': fac,
        'electricity_duty': electricity_duty,
        'tax_on_sale': tax_on_sale
    }

def get_billing_rates():
    # Load the rates from the JSON file
    with open('rates.json', 'r') as file:
        rates = json.load(file)
    return rates
