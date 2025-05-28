import pandas as pd
import re
import math
def extract_numeric_metrics(file_path):
    xls = pd.ExcelFile(file_path)
    data = []

    for sheet_name in xls.sheet_names:
        df = xls.parse(sheet_name)
        for i, row in df.iterrows():
            try:
                label = str(row[0]).strip()
                value = row[2]
                unit = row[3] if len(row) > 3 else ""
                if isinstance(value, (int, float)):
                    data.append((label, float(value), unit, sheet_name))
                elif isinstance(value, str) and re.search(r'\d', value):
                    val_clean = float(re.sub(r"[^\d.]", "", value.replace(",", "")))
                    data.append((label, val_clean, unit, sheet_name))
            except:
                continue

    return pd.DataFrame(data, columns=["Label", "Value", "Unit", "Sheet"])

def forecast_pu_consumption(base_value, base_year, cagr, years):
    """Forecast using CAGR over multiple years from base_value."""
    if base_value is None or math.isnan(base_value):
        raise ValueError("Base value is NaN or None — cannot forecast.")

    data = []
    for i in range(years + 1):
        year = base_year + i
        value = base_value * ((1 + cagr) ** i)
        data.append((f"FY{str(year)[-2:]}", round(value)))
    return pd.DataFrame(data, columns=["Year", "Forecast Value"])