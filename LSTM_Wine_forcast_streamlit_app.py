# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="Fine Wine Forecasting App",
    page_icon="🍷",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Data Loading and Caching ---
# Determine the absolute path to the CSV file
# Assuming the script is run from the root directory where 'Top37.csv' is located
# If the CSV is in a 'data' subdirectory, use 'data/Top37.csv'
script_dir = os.path.dirname(os.path.abspath(__file__))
# Try relative path first, then potentially absolute if needed
relative_path = "Chiaki_wine_prediction/Top37.csv"
absolute_path = os.path.join(script_dir, relative_path)

# Check if the file exists at the relative path
if not os.path.exists(relative_path):
    st.error(f"Error: Data file not found at relative path: {relative_path}")
    st.error(f"Attempting absolute path: {absolute_path}")
    if not os.path.exists(absolute_path):
        st.error(f"Error: Data file not found at absolute path either: {absolute_path}")
        st.stop() # Stop execution if data file is not found
    else:
        path = absolute_path
else:
    path = relative_path

st.sidebar.success("Select a page above to explore.")


@st.cache_data # Cache the data loading and preprocessing
def load_data(file_path):
    try:
        df = pd.read_csv(file_path, delimiter=",")

        # Convert DateTime column to date format - handle potential parsing errors
        try:
            df['Datetime'] = pd.to_datetime(df['Datetime'], format='mixed', errors='raise') # Raise error if parsing fails
        except Exception as e:
            st.warning(f"Could not automatically infer Datetime format: {e}. Trying manual parsing...")
            # Add more robust parsing if needed, e.g., specify format
            try:
                 df['Datetime'] = pd.to_datetime(df['Datetime'], errors='coerce') # Coerce errors to NaT
                 if df['Datetime'].isnull().any():
                     st.error("Some dates could not be parsed. Check the 'Datetime' column format.")
                     # Optionally drop rows with NaT or handle differently
                     # df = df.dropna(subset=['Datetime'])
            except Exception as final_e:
                 st.error(f"Failed to parse Datetime column even manually: {final_e}")
                 return None, None, None # Indicate failure

        # Set Datetime as index
        df = df.set_index('Datetime')

        # Convert all other columns to numeric format
        numeric_columns = df.columns
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Forward fill missing values AFTER numeric conversion
        df = df.ffill()

        # Handle potential remaining NaNs at the beginning
        df = df.bfill() # Backfill to handle NaNs at the very start

        # Check if all NaNs are handled
        if df.isna().sum().sum() > 0:
             st.warning("NaN values remain after ffill and bfill. Check input data.")
             st.write("NaN counts per column:")
             st.write(df.isna().sum())

        # Calculate percentage returns (monthly)
        returns_df = df.pct_change().dropna()

        wine_cols = df.columns.tolist()

        return df, returns_df, wine_cols
    except FileNotFoundError:
        st.error(f"Error: Data file not found at {file_path}")
        return None, None, None
    except Exception as e:
        st.error(f"An error occurred during data loading: {e}")
        import traceback
        st.error(traceback.format_exc())
        return None, None, None


# --- Load Data ---
df_original, df_returns, wine_columns = load_data(path)

if df_original is None:
    st.error("Failed to load or process data. Please check the file and its format.")
    st.stop()


# --- Store data in session state for access in other pages ---
# This is an alternative to importing the load_data function in each page
st.session_state['df_original'] = df_original
st.session_state['df_returns'] = df_returns
st.session_state['wine_columns'] = wine_columns


# --- Main Page Content ---
st.title("🍷 Fine Wine Analysis and Forecasting")

st.markdown("""
Welcome to the Fine Wine Analysis and Forecasting application.
This app demonstrates various time series analysis techniques applied to fine wine data.

**Navigate through the pages using the sidebar:**

1.  **EDA:** Explore the raw price trends, log prices, returns, time-series decomposition, and returns distributions.
2.  **HS VaR:** Calculate Historical Simulation Value at Risk for different fine wines and investment parameters.
3.  **ARIMA & SARIMA:** View predictions and evaluations using ARIMA and SARIMA models.
4.  **LSTM & DeepVaR:** Explore forecasts using Long Short-Term Memory networks and compare Historical VaR with DeepVaR estimates.

*Data Source: Top 37 Most Expensive Fine Wines (Monthly Prices: USD)*
""")

# st.header("Original Data Sample")
# st.dataframe(df_original.head())

# st.header("Calculated Monthly Returns Sample")
# st.dataframe(df_returns.head())