# pages/3_ARIMA_SARIMA.py
import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings

# Suppress specific statsmodels warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning, ValueWarning
warnings.simplefilter('ignore', ConvergenceWarning)
warnings.simplefilter('ignore', ValueWarning)
warnings.simplefilter('ignore', UserWarning) # General user warnings often related to frequency

st.set_page_config(page_title="ARIMA/SARIMA Forecast", layout="wide")

st.title("ARIMA & SARIMA Price Change Forecast")

# --- Access data from session state ---
if 'df_returns' not in st.session_state or 'wine_columns' not in st.session_state:
    st.error("Data not loaded. Please go back to the main page.")
    st.stop()

returns = st.session_state['df_returns']
wine_columns = st.session_state['wine_columns']

# --- User Inputs ---
st.sidebar.header("Model Options")
selected_wine_arima = st.sidebar.selectbox(
    "Select Wine",
    wine_columns,
    key="arima_wine_select"
)

forecast_horizon_arima = st.sidebar.slider(
    "Forecast Months",
    min_value=1,
    max_value=12, # Max forecast horizon
    value=12, # Default
    step=1,
    key="arima_horizon"
)

# --- Model Fitting and Prediction (Cached per Wine) ---
@st.cache_data(show_spinner=f"Running ARIMA/SARIMA for selected wine...")
def run_arima_sarima_models(wine_name, _returns_df): # Use _returns_df to ensure cache updates if data changes
    wine_returns = _returns_df[wine_name].copy().dropna()
    wine_returns.index = pd.to_datetime(wine_returns.index) # Ensure datetime index
    wine_returns = wine_returns.asfreq('MS', method='ffill') # Ensure monthly frequency

    results_dict = {'original': wine_returns}
    evaluation_metrics = {'ARIMA': {}, 'SARIMA': {}}
    min_data_points = 24 + 12 # Need enough for initial fit and test set

    if len(wine_returns) < min_data_points:
        st.warning(f"Not enough data ({len(wine_returns)} points) for reliable ARIMA/SARIMA fitting and evaluation for {wine_name}. Minimum required: {min_data_points}.")
        results_dict['error'] = "Insufficient data"
        return results_dict, evaluation_metrics

    # --- Fit Final Models on Full Data ---
    try:
        # ARIMA
        arima_model = ARIMA(wine_returns, order=(1, 1, 1)) # Simple order, adjust if needed
        arima_fit = arima_model.fit()
        arima_forecast_obj = arima_fit.get_forecast(steps=12) # Always forecast 12 for storage
        results_dict['arima_forecast'] = arima_forecast_obj.predicted_mean
        results_dict['arima_ci'] = arima_forecast_obj.conf_int()

        # SARIMA
        sarima_model = SARIMAX(wine_returns, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)) # Example order
        sarima_fit = sarima_model.fit(disp=False)
        sarima_forecast_obj = sarima_fit.get_forecast(steps=12)
        results_dict['sarima_forecast'] = sarima_forecast_obj.predicted_mean
        results_dict['sarima_ci'] = sarima_forecast_obj.conf_int()

    except Exception as e:
        st.error(f"Error fitting final models for {wine_name}: {e}")
        results_dict['error'] = f"Fitting error: {e}"
        return results_dict, evaluation_metrics # Return partial results if error occurs


    # --- Fit Models on Train Set for Evaluation ---
    try:
        train = wine_returns[:-12]
        test = wine_returns[-12:]

        # ARIMA Eval
        arima_model_eval = ARIMA(train, order=(1, 1, 1))
        arima_fit_eval = arima_model_eval.fit()
        arima_forecast_eval = arima_fit_eval.get_forecast(steps=12).predicted_mean
        arima_forecast_eval.index = test.index # Align index for comparison

        # SARIMA Eval
        sarima_model_eval = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
        sarima_fit_eval = sarima_model_eval.fit(disp=False)
        sarima_forecast_eval = sarima_fit_eval.get_forecast(steps=12).predicted_mean
        sarima_forecast_eval.index = test.index

        # Calculate Metrics
        def calculate_metrics(y_true, y_pred):
            mae = mean_absolute_error(y_true, y_pred)
            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            # MAPE requires handling zeros
            mask = y_true != 0
            mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if np.any(mask) else np.inf
            return {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'MAPE': mape}

        evaluation_metrics['ARIMA'] = calculate_metrics(test, arima_forecast_eval)
        evaluation_metrics['SARIMA'] = calculate_metrics(test, sarima_forecast_eval)
        results_dict['test_actual'] = test # Store test actuals for potential plotting

    except Exception as e:
        st.warning(f"Error during model evaluation for {wine_name}: {e}")
        evaluation_metrics['Error'] = f"Evaluation error: {e}"


    return results_dict, evaluation_metrics

# --- Run Models and Display Results ---
st.header(f"Forecast for: {selected_wine_arima}")

model_results, metrics = run_arima_sarima_models(selected_wine_arima, returns)

if 'error' in model_results:
    st.error(f"Could not generate forecast for {selected_wine_arima}. Reason: {model_results['error']}")
else:
    # --- Plot Forecast ---
    fig_forecast = go.Figure()

    # Original Data
    fig_forecast.add_trace(go.Scatter(
        x=model_results['original'].index,
        y=model_results['original'].values,
        mode='lines', name='Historical Price Change', line=dict(color='blue')
    ))

    # Forecast Horizon data points
    horizon = forecast_horizon_arima
    arima_future_dates = model_results['arima_forecast'].index[:horizon]
    sarima_future_dates = model_results['sarima_forecast'].index[:horizon] # Should be the same

    # ARIMA Forecast & CI
    arima_fc = model_results['arima_forecast'][:horizon]
    arima_ci = model_results['arima_ci'][:horizon]
    fig_forecast.add_trace(go.Scatter(
        x=arima_future_dates, y=arima_fc,
        mode='lines', name='ARIMA Forecast', line=dict(color='orange')
    ))
    fig_forecast.add_trace(go.Scatter(
        x=list(arima_future_dates) + list(arima_future_dates[::-1]),
        y=list(arima_ci.iloc[:, 1]) + list(arima_ci.iloc[:, 0][::-1]), # upper + lower reversed
        fill='toself', fillcolor='rgba(255,165,0,0.2)', line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip", showlegend=True, name='ARIMA 95% CI'
    ))

    # SARIMA Forecast & CI
    sarima_fc = model_results['sarima_forecast'][:horizon]
    sarima_ci = model_results['sarima_ci'][:horizon]
    fig_forecast.add_trace(go.Scatter(
        x=sarima_future_dates, y=sarima_fc,
        mode='lines', name='SARIMA Forecast', line=dict(color='green')
    ))
    fig_forecast.add_trace(go.Scatter(
        x=list(sarima_future_dates) + list(sarima_future_dates[::-1]),
        y=list(sarima_ci.iloc[:, 1]) + list(sarima_ci.iloc[:, 0][::-1]), # upper + lower reversed
        fill='toself', fillcolor='rgba(0,128,0,0.2)', line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip", showlegend=True, name='SARIMA 95% CI'
    ))

    fig_forecast.update_layout(
        title=f"{selected_wine_arima} - {horizon}-Month Price Change Forecast",
        xaxis_title="Date", yaxis_title="Monthly Price Change (%)",
        height=500, template='plotly_white', hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_forecast, use_container_width=True)


    # --- Display Evaluation Metrics ---
    st.subheader("Model Evaluation Metrics (on Test Set)")
    if 'ARIMA' in metrics and metrics['ARIMA']:
         metrics_df = pd.DataFrame(metrics).T # Transpose for better display
         metrics_df = metrics_df.map(lambda x: f"{x:.4f}" if isinstance(x, (int, float)) and x != np.inf else x) # Format numbers
         st.dataframe(metrics_df)
    elif 'Error' in metrics:
         st.warning(f"Evaluation could not be performed: {metrics['Error']}")
    else:
         st.info("Evaluation metrics are not available (e.g., due to insufficient data).")