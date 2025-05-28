import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

def load_data(file_path, sheet_name=None):
    if sheet_name:
        df = pd.read_excel(file_path, sheet_name=sheet_name)
    else:
        df = pd.read_excel(file_path)
    return df

# Example
data = load_data('market.xlsx')
print(data.head())

def preprocess_data(df, year_col, value_col):
    # Drop rows with missing year or value
    df_clean = df.dropna(subset=[year_col, value_col])
    # Ensure correct types
    df_clean[year_col] = df_clean[year_col].astype(int)
    df_clean[value_col] = df_clean[value_col].astype(float)
    # Sort by year ascending
    df_clean = df_clean.sort_values(by=year_col)
    return df_clean

# Example
data_clean = preprocess_data(data, year_col='Year', value_col='PU_Consumption')
print(data_clean)


def calculate_cagr(start_value, end_value, num_years):
    return (end_value / start_value) ** (1 / num_years) - 1

# Example usage:
start = data_clean['PU_Consumption'].iloc[0]
end = data_clean['PU_Consumption'].iloc[-1]
years = data_clean['Year'].iloc[-1] - data_clean['Year'].iloc[0]
cagr = calculate_cagr(start, end, years)
print(f"CAGR calculated from data: {cagr:.4f} or {cagr*100:.2f}% per year")


def fit_linear_regression(df, year_col, value_col):
    X = df[[year_col]].values
    y = df[value_col].values
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    mae = mean_absolute_error(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    print(f"Linear Regression MAE: {mae:.2f}, MSE: {mse:.2f}")
    return model

model = fit_linear_regression(data_clean, 'Year', 'PU_Consumption')


def forecast_demand(model, start_year, forecast_years):
    future_years = np.array([start_year + i for i in range(1, forecast_years + 1)]).reshape(-1, 1)
    forecast = model.predict(future_years)
    forecast = np.maximum(forecast, 0)  # no negative forecast
    forecast_df = pd.DataFrame({'Year': future_years.flatten(), 'Forecasted_Consumption': forecast})
    return forecast_df

# Example:
last_year = data_clean['Year'].max()
forecast_years = 10
forecast_df = forecast_demand(model, last_year, forecast_years)
print(forecast_df)




def plot_forecast(data, forecast, year_col='Year', value_col='PU_Consumption', forecast_col='Forecasted_Consumption'):
    plt.figure(figsize=(10, 6))
    plt.plot(data[year_col], data[value_col], marker='o', label='Historical Data')
    plt.plot(forecast[year_col], forecast[forecast_col], marker='x', linestyle='--', color='red', label='Forecast')
    plt.xlabel('Year')
    plt.ylabel('Consumption (tons)')
    plt.title('Market Demand Forecast')
    plt.legend()
    plt.grid(True)
    plt.show()

plot_forecast(data_clean, forecast_df)


def apply_scenario(forecast_df, base_cagr=None, optimistic_cagr=None):
    last_forecast = forecast_df['Forecasted_Consumption'].iloc[-1]

    if base_cagr:
        forecast_df['Base_Case'] = forecast_df['Year'].apply(
            lambda y: last_forecast * ((1 + base_cagr) ** (y - forecast_df['Year'].min()))
        )
    if optimistic_cagr:
        forecast_df['Optimistic_Case'] = forecast_df['Year'].apply(
            lambda y: last_forecast * ((1 + optimistic_cagr) ** (y - forecast_df['Year'].min()))
        )
    return forecast_df

# Example (with hypothetical CAGR values)
forecast_df = apply_scenario(forecast_df, base_cagr=0.11, optimistic_cagr=0.13)
print(forecast_df)


