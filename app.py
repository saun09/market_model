import streamlit as st
from forecast_utils import prepare_forecast_table
import matplotlib.pyplot as plt

st.set_page_config(page_title="Market Demand Forecast", layout="centered")

st.title(" AI-powered Market Forecasting Tool")
st.write("Use assumptions or real data to project demand/value for the next 10 years.")

# Sidebar Inputs
base_value = st.number_input("Current Year Demand / Value (e.g. 17500 Cr)", min_value=0.0, value=17500.0)
base_year = st.number_input("Base Year (e.g. 2023)", value=2023)
growth_rate = st.slider("Expected Annual Growth Rate (%)", min_value=0.0, max_value=25.0, value=7.0)
forecast_years = st.slider("Forecast Period (Years)", min_value=1, max_value=20, value=10)

if st.button("Generate Forecast"):
    df_forecast = prepare_forecast_table(base_year, base_value, forecast_years, growth_rate / 100)
    st.write("### 📊 Forecast Table")
    st.dataframe(df_forecast)

    # Plot
    st.write("### 📉 Forecast Plot")
    fig, ax = plt.subplots()
    ax.plot(df_forecast['Year'], df_forecast['Forecast'], marker='o')
    ax.set_xlabel("Year")
    ax.set_ylabel("Forecasted Value")
    ax.set_title("Market Demand / Value Forecast")
    st.pyplot(fig)

st.write("---")
st.caption("Prototype by Saundarya for internship project.")
