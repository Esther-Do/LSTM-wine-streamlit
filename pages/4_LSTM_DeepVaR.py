# pages/4_LSTM_DeepVaR.py
import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model # To load pre-trained models
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import joblib # To load pre-saved scalers
import calendar
import os
import warnings

# Suppress TensorFlow warnings if desired
# tf.get_logger().setLevel('ERROR')
# warnings.filterwarnings('ignore')

st.set_page_config(page_title="LSTM Forecast & DeepVaR", layout="wide")

st.title("LSTM Forecast and DeepVaR Analysis")

# --- Configuration ---
MODEL_DIR = "saved_models" # Directory to load models/scalers from
LAG = 12 # Must match the lag used during training

# --- Helper Functions ---
def sanitize_filename(name):
    """Removes characters problematic for filenames."""
    name = name.replace(',', '').replace('\'', '').replace('&', 'and')
    return "_".join(name.split())[:50] # Limit length

# Function to calculate 95% confidence interval (adjust if needed)
def calculate_95ci(predictions):
    """ Placeholder for CI calculation - robust calculation is complex """
    if predictions is None or len(predictions) == 0:
        return np.array([]), np.array([])
    std_dev = np.std(predictions) # Simple std dev - might underestimate uncertainty
    ci_lower = predictions - 1.96 * std_dev
    ci_upper = predictions + 1.96 * std_dev
    return ci_lower, ci_upper

def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def generate_future_month_strings(last_date, num_months):
    """Generates formatted month strings for future dates."""
    if isinstance(last_date, (str, np.datetime64)):
        last_date = pd.to_datetime(last_date)

    future_months = []
    for i in range(1, num_months + 1):
        future_date = last_date + pd.DateOffset(months=i)
        month_name = calendar.month_abbr[future_date.month]
        future_months.append(f"{month_name} {future_date.year}")
    return future_months

# Function to calculate Historical VaR
def hs_var_calc(returns_series, alpha=95):
    q = 100 - alpha
    var_percent = -np.percentile(returns_series.dropna(), q)
    return np.round(var_percent, 4)

# --- Access data from session state ---
if 'df_returns' not in st.session_state or 'wine_columns' not in st.session_state:
    st.error("Data not loaded. Please go back to the main page.")
    st.stop()

returns = st.session_state['df_returns']
wine_columns = st.session_state['wine_columns']
df_original_index = st.session_state['df_original'].index # Get original datetime index

# --- User Inputs ---
st.sidebar.header("LSTM/DeepVaR Options")
selected_wine_lstm = st.sidebar.selectbox(
    "Select Wine",
    wine_columns,
    key="lstm_wine_select"
)

forecast_horizon_lstm = st.sidebar.radio(
    "Display Horizon",
    ('3 Months', '6 Months', '12 Months'),
    index=2, # Default to 12 months
    key="lstm_horizon"
)

investment_lstm = st.sidebar.slider(
    "Investment Amount ($)",
    min_value=100, max_value=10000, value=1000, step=100,
    key="lstm_investment"
)

# --- Load Model and Scaler (Cached per Wine) ---
@st.cache_resource(show_spinner="Loading LSTM model and scaler...")
def load_lstm_resource(wine_name):
    """Loads the pre-trained Keras model and scaler."""
    sanitized_name = sanitize_filename(wine_name)
    model_path = os.path.join(MODEL_DIR, f"model_{sanitized_name}.keras")
    scaler_path = os.path.join(MODEL_DIR, f"scaler_{sanitized_name}.joblib")

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        st.error(f"Model or scaler not found for {wine_name} in '{MODEL_DIR}'. Expected paths:\n- {model_path}\n- {scaler_path}")
        return None, None

    try:
        model = load_model(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler
    except Exception as e:
        st.error(f"Error loading model/scaler for {wine_name}: {e}")
        return None, None

# --- Generate Predictions and VaR (Cached per Wine/Investment) ---
@st.cache_data(show_spinner="Generating LSTM predictions and VaR...")
def get_lstm_predictions_and_var(wine_name, _returns_df, _investment_amount):
    """Generates LSTM predictions and calculates VaR metrics."""
    model, scaler = load_lstm_resource(wine_name)
    if model is None or scaler is None:
        return {"error": "Model/Scaler not loaded"}

    wine_returns = _returns_df[wine_name].copy().dropna()
    historical_var_perc = hs_var_calc(wine_returns, alpha=95) # 95% historical VaR

    # Scale the full time series
    time_series_scaled = scaler.transform(wine_returns.values.reshape(-1, 1))

    # Prepare sequence for future prediction
    last_sequence = time_series_scaled[-LAG:]
    if len(last_sequence) < LAG:
         return {"error": f"Insufficient data length ({len(wine_returns)}) to create lag sequence of {LAG}"}

    # Predict Future
    future_predictions_scaled = []
    current_sequence = last_sequence.reshape((1, LAG, 1))

    for _ in range(12): # Predict 12 steps ahead
        next_pred_scaled = model.predict(current_sequence, verbose=0)[0, 0]
        future_predictions_scaled.append(next_pred_scaled)
        # Update sequence: remove first element, append prediction
        new_sequence_scaled = np.append(current_sequence[0, 1:, 0], next_pred_scaled).reshape((1, LAG, 1))
        current_sequence = new_sequence_scaled

    # Inverse transform future predictions
    future_predictions_inverse = scaler.inverse_transform(np.array(future_predictions_scaled).reshape(-1, 1))

    # Calculate Deep VaR
    deep_var_values = []
    deep_var_amounts = []
    # Simple DeepVaR: Use the 5th percentile of *predicted* future returns
    # More complex methods exist (e.g., simulating from predicted distribution)
    # Here, we'll just take the 5th percentile of the 12 predicted returns
    if len(future_predictions_inverse) > 0:
        pred_returns_flat = future_predictions_inverse.flatten()
        for i in range(1, 13):
             # Use the predicted returns up to month i
             current_preds = pred_returns_flat[:i]
             if len(current_preds) > 0:
                  # Calculate VaR based on the model's predicted returns for that month
                  # This is a simplified approach. A more robust way involves quantile regression
                  # or simulating paths based on predicted volatility.
                  # Here we use the predicted value directly as a point estimate of loss,
                  # or calculate percentile from *all* future predictions.
                  # Let's use the percentile of all 12 future preds as a proxy.
                  var_value_deep = -np.percentile(pred_returns_flat, 5) # 5th percentile of the 12 predictions
             else:
                  var_value_deep = np.nan

             deep_var_values.append(var_value_deep if not np.isnan(var_value_deep) else 0) # Handle potential NaN
             deep_var_amounts.append((var_value_deep * _investment_amount) if not np.isnan(var_value_deep) else 0)


    # Prepare results for plotting test performance (optional but good practice)
    def create_lagged_features(data, lag):
        X, y = [], []
        for i in range(len(data) - lag):
            X.append(data[i:i+lag])
            y.append(data[i+lag])
        return np.array(X), np.array(y)

    X, y = create_lagged_features(time_series_scaled, LAG)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    train_size = int(0.80 * len(X)) # Assuming 80/20 split used in training
    X_test = X[train_size:]
    y_test = y[train_size:]

    y_pred_lstm_scaled = model.predict(X_test, verbose=0)
    y_pred_lstm_inverse = scaler.inverse_transform(y_pred_lstm_scaled)
    y_test_inverse = scaler.inverse_transform(y_test)
    test_rmse = root_mean_squared_error(y_test_inverse, y_pred_lstm_inverse)
    test_ci_lower, test_ci_upper = calculate_95ci(y_pred_lstm_inverse) # Simple CI


    return {
        "original_returns": wine_returns,
        "y_test_inverse": y_test_inverse,
        "y_pred_lstm_inverse": y_pred_lstm_inverse,
        "test_rmse": test_rmse,
        "test_ci_lower": test_ci_lower,
        "test_ci_upper": test_ci_upper,
        "future_predictions": future_predictions_inverse,
        "historical_var_perc": historical_var_perc,
        "deep_var_values": deep_var_values,
        "deep_var_amounts": deep_var_amounts,
        "train_split_index": train_size + LAG # Index in original returns where test starts
    }

# --- Run Model and Display Results ---
st.header(f"LSTM Forecast & DeepVaR for: {selected_wine_lstm}")

results = get_lstm_predictions_and_var(selected_wine_lstm, returns, investment_lstm)

if "error" in results:
    st.error(f"Could not generate forecast for {selected_wine_lstm}. Reason: {results['error']}")
else:
    # Determine display months
    horizon_map = {'3 Months': 3, '6 Months': 6, '12 Months': 12}
    display_months = horizon_map[forecast_horizon_lstm]

    # Get dates
    original_dates = results['original_returns'].index
    test_start_index_in_orig = results['train_split_index']
    test_dates = original_dates[test_start_index_in_orig : test_start_index_in_orig + len(results['y_test_inverse'])]
    last_historical_date = original_dates[-1]
    future_month_strings = generate_future_month_strings(last_historical_date, 12)

    # --- Plot 1: Historical Performance ---
    st.subheader("LSTM Model Performance on Test Data")
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Scatter(x=original_dates, y=results['original_returns'], mode='lines', name='Historical Returns', line=dict(color='blue')))
    fig_hist.add_trace(go.Scatter(x=test_dates, y=results['y_test_inverse'].flatten(), mode='lines', name='True Test Returns', line=dict(color='green')))
    fig_hist.add_trace(go.Scatter(x=test_dates, y=results['y_pred_lstm_inverse'].flatten(), mode='lines', name='LSTM Test Predictions', line=dict(color='red')))
    # Add Test CI
    fig_hist.add_trace(go.Scatter(x=test_dates, y=results['test_ci_upper'].flatten(), mode='lines', line=dict(width=0), showlegend=False))
    fig_hist.add_trace(go.Scatter(x=test_dates, y=results['test_ci_lower'].flatten(), mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(255,0,0,0.1)', name='95% CI (Test - Simple)'))

    fig_hist.update_layout(
        title=f"LSTM Test Performance (RMSE: {results['test_rmse']:.3})",
        xaxis_title='Date', yaxis_title='Monthly Return', height=400, template='plotly_white', hovermode='x unified'
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    # --- Plot 2: Future Predictions ---
    st.subheader(f"LSTM Future Return Predictions ({forecast_horizon_lstm})")
    fig_future = go.Figure()
    future_preds_display = results['future_predictions'][:display_months]
    future_months_display = future_month_strings[:display_months]
    # Simple CI for future preds
    future_ci_lower, future_ci_upper = calculate_95ci(future_preds_display)

    fig_future.add_trace(go.Scatter(x=future_months_display, y=future_preds_display.flatten(), mode='lines+markers', name='Predicted Future Return', line=dict(color='purple')))
    # Add Future CI
    fig_future.add_trace(go.Scatter(x=future_months_display, y=future_ci_upper.flatten(), mode='lines', line=dict(width=0), showlegend=False))
    fig_future.add_trace(go.Scatter(x=future_months_display, y=future_ci_lower.flatten(), mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(128,0,128,0.1)', name='95% CI (Future - Simple)'))

    fig_future.update_layout(
        title=f"Predicted Future Monthly Returns ({forecast_horizon_lstm})",
        xaxis_title='Month', yaxis_title='Predicted Monthly Return', height=400, template='plotly_white', hovermode='x unified'
    )
    st.plotly_chart(fig_future, use_container_width=True)

    # --- Plot 3: VaR Comparison ---
    st.subheader(f"Value at Risk (VaR) Comparison for ${investment_lstm:,.0f} Investment")
    fig_var = go.Figure()
    hist_var_amount = results['historical_var_perc'] * investment_lstm
    deep_var_amounts_display = results['deep_var_amounts'][:display_months]

    fig_var.add_trace(go.Scatter(x=future_months_display, y=[hist_var_amount] * display_months, mode='lines', name=f'Historical VaR (${hist_var_amount:,.2f})', line=dict(color='red', dash='dash')))
    fig_var.add_trace(go.Scatter(x=future_months_display, y=deep_var_amounts_display, mode='lines+markers', name='Deep VaR (LSTM Based)', line=dict(color='blue')))

    fig_var.update_layout(
        title=f"Historical VaR vs. DeepVaR (95% Confidence, ${investment_lstm:,.0f} Investment)",
        xaxis_title='Forecast Month', yaxis_title='VaR Amount ($) - Potential Loss', height=400, template='plotly_white', hovermode='x unified'
    )
    st.plotly_chart(fig_var, use_container_width=True)

    # --- Table: VaR Comparison ---
    st.subheader(f"VaR Comparison Data ({forecast_horizon_lstm})")
    var_comp_data = {
        'Month': future_months_display,
        'Hist. VaR (%)': [f"{results['historical_var_perc']:.2%}"] * display_months,
        'Hist. VaR ($)': [f"${hist_var_amount:,.2f}"] * display_months,
        'Deep VaR (%)': [f"{v:.2%}" for v in results['deep_var_values'][:display_months]],
        'Deep VaR ($)': [f"${a:,.2f}" for a in deep_var_amounts_display]
    }
    st.dataframe(pd.DataFrame(var_comp_data))