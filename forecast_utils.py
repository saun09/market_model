import pandas as pd
import numpy as np

def apply_cagr_forecast(base_value, years, growth_rate):
    """Forecasts values for future years using CAGR."""
    return [base_value * ((1 + growth_rate) ** i) for i in range(1, years + 1)]

def prepare_forecast_table(base_year, base_value, years, growth_rate):
    """Generates a forecast DataFrame."""
    forecast_values = apply_cagr_forecast(base_value, years, growth_rate)
    future_years = list(range(base_year + 1, base_year + years + 1))
    return pd.DataFrame({'Year': future_years, 'Forecast': forecast_values})

def load_sample_data(filepath):
    """Optional loader for CSV with year-wise data."""
    df = pd.read_csv(filepath)
    return df
