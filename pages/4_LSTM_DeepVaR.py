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
tf.get_logger().setLevel('ERROR')
warnings.filterwarnings('ignore')

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

# Function to calculate 95% confidence interval
def calculate_95ci(predictions):
    """ Calculate 95% confidence interval """
    if predictions is None or len(predictions) == 0:
        return np.array([]), np.array([])
    std_dev = np.std(predictions)
    ci_lower = predictions - 1.96 * std_dev
    ci_upper = predictions + 1.96 * std_dev
    return ci_lower, ci_upper

def calculate_99ci(predictions):
    """Calculate 99% confidence interval"""
    if predictions is None or len(predictions) == 0:
        return np.array([]), np.array([])
    std_dev = np.std(predictions)
    ci_lower = predictions - 2.576 * std_dev
    ci_upper = predictions + 2.576 * std_dev
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
def hs_var_calc(returns_series):
    """Calculate Historical VaR for both 95% and 99% confidence levels"""
    var_95 = -np.percentile(returns_series.dropna(), 5)  # 95% VaR
    var_99 = -np.percentile(returns_series.dropna(), 1)  # 99% VaR
    return np.round(var_95, 4), np.round(var_99, 4)

# --- Access data from session state ---
if 'df_returns' not in st.session_state or 'wine_columns' not in st.session_state:
    st.error("Data not loaded. Please go back to the main page.")
    st.stop()

returns = st.session_state['df_returns']
wine_columns = st.session_state['wine_columns']
df_original = st.session_state['df_original']
df_original_index = df_original.index # Get original datetime index

# --- User Inputs ---
st.sidebar.header("DeepVaR Options")
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

# Function to prepare data for LSTM prediction
def prepare_lstm_data(time_series, lag=LAG):
    """Prepares data in the format expected by LSTM model."""
    X, y = [], []
    for i in range(len(time_series) - lag):
        X.append(time_series[i:i+lag])
        y.append(time_series[i+lag])
    return np.array(X), np.array(y)

# Function to make future predictions
# Note: Using _scaler with underscore to avoid hashing issues with StandardScaler
def generate_predictions(wine_name, time_series, model, _scaler, lag=LAG):
    """Generates historical and future predictions using the LSTM model."""
    # Use _scaler instead of scaler in the function signature to avoid hashing issues
    scaler = _scaler
    
    # Prepare data
    X, y = prepare_lstm_data(time_series, lag)
    
    # Split data into train/test (80/20)
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Make predictions on test data
    y_pred_lstm = model.predict(X_test)
    
    # Inverse transform predictions and actual values
    y_test_inverse = scaler.inverse_transform(y_test.reshape(-1, 1))
    y_pred_lstm_inverse = scaler.inverse_transform(y_pred_lstm)
    
    # Calculate RMSE
    rmse = root_mean_squared_error(y_test_inverse, y_pred_lstm_inverse)
    
    # Calculate confidence intervals for test predictions
    ci_lower_95, ci_upper_95 = calculate_95ci(y_pred_lstm_inverse)
    ci_lower_99, ci_upper_99 = calculate_99ci(y_pred_lstm_inverse)

    # Generate future predictions
    last_sequence = time_series[-lag:].reshape(1, lag, 1)
    future_predictions = []
    current_sequence = last_sequence.copy()
    
    # Predict next 12 months
    for _ in range(12):
        next_pred = model.predict(current_sequence, verbose=0)
        future_predictions.append(next_pred[0, 0])
        current_sequence = np.roll(current_sequence, -1)
        current_sequence[0, -1, 0] = next_pred[0, 0]
    
    # Convert future predictions array and inverse transform
    future_predictions = np.array(future_predictions).reshape(-1, 1)
    future_predictions_inverse = scaler.inverse_transform(future_predictions)
    
    # Now calculate confidence intervals with the inverse transformed predictions
    future_lower_95, future_upper_95 = calculate_95ci(future_predictions_inverse)
    future_lower_99, future_upper_99 = calculate_99ci(future_predictions_inverse)
    
    # Calculate Historical VaR
    historical_var_95, historical_var_99 = hs_var_calc(returns[wine_name])
    
    # Calculate DeepVaR using future predictions
    deep_var_values_95 = []
    deep_var_amounts_95 = []
    deep_var_values_99 = []
    deep_var_amounts_99 = []
    
    for i in range(len(future_predictions_inverse)):
        # Calculate 95% VaR
        potential_loss_95 = (future_predictions_inverse[i] - future_lower_95[i]) / future_predictions_inverse[i]
        deep_var_values_95.append(float(potential_loss_95))
        deep_var_amounts_95.append(float(potential_loss_95 * 1000))
        
        # Calculate 99% VaR
        potential_loss_99 = (future_predictions_inverse[i] - future_lower_99[i]) / future_predictions_inverse[i]
        deep_var_values_99.append(float(potential_loss_99))
        deep_var_amounts_99.append(float(potential_loss_99 * 1000))

    return {
        'time_series': time_series,
        'lag': lag,
        'train_size': train_size,
        'y_test_inverse': y_test_inverse,
        'y_pred_lstm_inverse': y_pred_lstm_inverse,
        'ci_lower_95': ci_lower_95,
        'ci_upper_95': ci_upper_95,
        'ci_lower_99': future_lower_99,
        'ci_upper_99': future_upper_99,
        'rmse': rmse,
        'future_predictions': future_predictions_inverse,
        'future_lower_95': future_lower_95,
        'future_upper_95': future_upper_95,
        'future_lower_99': future_lower_99,
        'future_upper_99': future_upper_99,
        'historical_var_95': historical_var_95,
        'historical_var_99': historical_var_99,
        'deep_var_values_95': deep_var_values_95,
        'deep_var_amounts_95': deep_var_amounts_95,
        'deep_var_values_99': deep_var_values_99,
        'deep_var_amounts_99': deep_var_amounts_99
    }

# Function to plot the selected wine's forecast and VaR
def plot_wine_forecast_and_var(wine, horizon, investment):
    # Load model and scaler
    model, scaler = load_lstm_resource(wine)
    
    if model is None or scaler is None:
        st.error(f"Could not load model or scaler for {wine}")
        return None, None, None, None
    
    # Get the time series data
    time_series = returns[wine].values.reshape(-1, 1)
    
    # Scale the data
    scaled_data = scaler.transform(time_series)
    
    # Generate predictions
    model_data = generate_predictions(wine, scaled_data, model, scaler)
    
    # Get the data
    time_series = model_data['time_series']
    lag = model_data['lag']
    train_size = model_data['train_size']
    y_test_inverse = model_data['y_test_inverse']
    y_pred_lstm_inverse = model_data['y_pred_lstm_inverse']
    ci_lower = model_data['ci_lower_95']
    ci_upper = model_data['ci_upper_95']
    rmse = model_data['rmse']
    future_predictions = model_data['future_predictions']
    future_lower_95 = model_data['future_lower_95']
    future_upper_95 = model_data['future_upper_95']
    historical_var_95 = model_data['historical_var_95']
    deep_var_values_95 = model_data['deep_var_values_95']
    deep_var_amounts_95 = model_data['deep_var_amounts_95']
    future_lower_99 = model_data['future_lower_99']
    future_upper_99 = model_data['future_upper_99']
    historical_var_99 = model_data['historical_var_99']
    deep_var_values_99 = model_data['deep_var_values_99']
    deep_var_amounts_99 = model_data['deep_var_amounts_99']
    
    # Calculate indices for train and test data
    train_start_idx = lag
    train_end_idx = train_size + lag
    test_end_idx = train_end_idx + len(y_test_inverse)
    
    # Ensure we don't go out of bounds
    max_idx = len(df_original_index)
    train_start_idx = min(train_start_idx, max_idx-1)
    train_end_idx = min(train_end_idx, max_idx)
    test_end_idx = min(test_end_idx, max_idx)
    
    # Get the dates for historical data
    historical_dates = df_original_index[-len(time_series):]
    train_dates = df_original_index[train_start_idx:train_end_idx]
    test_dates = df_original_index[train_end_idx:test_end_idx]
    
    # Get the last date in the dataset
    last_date = df_original_index[-1]
    
    # Determine how many months to display based on selected horizon
    if horizon == '3 Months':
        display_months = 3
    elif horizon == '6 Months':
        display_months = 6
    else:  # 12 Months
        display_months = 12
    
    # Generate future month strings
    future_month_strings = generate_future_month_strings(last_date, 12)
    
    # Create Plotly figure for historical performance
    fig = go.Figure()
    
    # Add actual data
    fig.add_trace(go.Scatter(
        x=historical_dates, 
        y=scaler.inverse_transform(time_series.reshape(-1, 1)).flatten(),
        mode='lines',
        name='Historical Data',
        line=dict(color='blue')
    ))
    
    # Add true test values
    fig.add_trace(go.Scatter(
        x=test_dates, 
        y=y_test_inverse.flatten(),
        mode='lines',
        name='True Test Values',
        line=dict(color='green')
    ))
    
    # Add test predictions
    fig.add_trace(go.Scatter(
        x=test_dates, 
        y=y_pred_lstm_inverse.flatten(),
        mode='lines',
        name='Predictions Values',
        line=dict(color='red')
    ))
    
    # Add test confidence interval - upper bound
    fig.add_trace(go.Scatter(
        x=test_dates,
        y=ci_upper.flatten(),
        mode='lines',
        line=dict(width=0),
        showlegend=False
    ))
    
    # Add test confidence interval - lower bound
    fig.add_trace(go.Scatter(
        x=test_dates,
        y=ci_lower.flatten(),
        mode='lines',
        line=dict(width=0),
        fill='tonexty',
        fillcolor='rgba(255,165,0,0.2)',
        name='95% CI (Test)'
    ))
    
    # Update historical figure layout
    fig.update_layout(
        title=dict(
            text=f'{wine} <br> LSTM Historical Performance. RMSE: {rmse:.3f}',
            x=0.5,  # Center the title horizontally
            xanchor='center'
        ),
        xaxis_title='Date',
        yaxis_title='Returns',
        legend_title='Legend',
        height=400,
        template='plotly_white',
        hovermode='x unified'
    )
    
    # Create figure for future predictions
    future_fig = go.Figure()
    
    # Add future predictions - only show the selected number of months
    display_month_strings = future_month_strings[:display_months]
    display_predictions = future_predictions[:display_months]
    display_lower_95 = future_lower_95[:display_months]
    display_upper_95 = future_upper_95[:display_months]
    display_lower_99 = future_lower_99[:display_months]
    display_upper_99 = future_upper_99[:display_months]
    
    # Add future predictions line
    future_fig.add_trace(go.Scatter(
        x=display_month_strings, 
        y=display_predictions.flatten(),
        mode='lines+markers',
        name=f'Future Predictions ({horizon})',
        line=dict(color='purple'),
        marker=dict(size=8)
    ))
    
    # Add future 95 confidence interval - upper bound
    future_fig.add_trace(go.Scatter(
        x=display_month_strings,
        y=display_upper_95.flatten(),
        mode='lines',
        line=dict(width=0),
        showlegend=False
    ))
    
    # Add future 95 confidence interval - lower bound
    future_fig.add_trace(go.Scatter(
        x=display_month_strings,
        y=display_lower_95.flatten(),
        mode='lines',
        line=dict(width=0),
        fill='tonexty',
        fillcolor='rgba(128,0,128,0.2)',
        name=f'95% CI ({horizon})'
    ))
    
      # Add future 99 confidence interval - upper bound
    future_fig.add_trace(go.Scatter(
        x=display_month_strings,
        y=display_upper_99.flatten(),
        mode='lines',
        line=dict(width=0),
        showlegend=False
    ))
    
    # Add future 99 confidence interval - lower bound
    future_fig.add_trace(go.Scatter(
        x=display_month_strings,
        y=display_lower_99.flatten(),
        mode='lines',
        line=dict(width=0),
        fill='tonexty',
        fillcolor='rgba(128,0,128,0.2)',
        name=f'95% CI ({horizon})'
    ))

    # Update future figure layout
    future_fig.update_layout(
        title=dict(text=f'Future Predictions ({horizon})',
                   x=0.5,  # Center the title horizontally
                   xanchor='center'),
        xaxis_title='Month',
        yaxis_title='Returns',
        legend_title='Legend',
        height=400,
        template='plotly_white',
        hovermode='x unified'
    )
    
    # Create VaR figure
    var_fig = go.Figure()
    
    # Inside plot_wine_forecast_and_var function, update the VaR figure creation:
    # Calculate VaR amounts
    historical_var_amount_95 = investment * model_data['historical_var_95']
    historical_var_amount_99 = investment * model_data['historical_var_99']
    
    # Scale Deep VaR amounts
    scaled_deep_var_amounts_95 = [amount * (investment / 1000) for amount in model_data['deep_var_amounts_95']]
    scaled_deep_var_amounts_99 = [amount * (investment / 1000) for amount in model_data['deep_var_amounts_99']]
    
    pnl_values = [investment * pred for pred in future_predictions[:display_months].flatten()]

    # Add PnL line
    var_fig.add_trace(go.Scatter(
        x=display_month_strings,
        y=pnl_values,
        mode='lines+markers',
        name='Expected PnL',
        line=dict(color='green'),
        marker=dict(size=8)
    ))

    # Add VaR lines to figure
    var_fig.add_trace(go.Scatter(
        x=display_month_strings,
        y=[historical_var_amount_95] * len(display_month_strings),
        mode='lines',
        name='HS VaR 95%',
        line=dict(color='orange', dash='dash')
    ))
    
    var_fig.add_trace(go.Scatter(
        x=display_month_strings,
        y=[historical_var_amount_99] * len(display_month_strings),
        mode='lines',
        name='HS VaR 99%',
        line=dict(color='red', dash='dash')
    ))
    
    var_fig.add_trace(go.Scatter(
        x=display_month_strings,
        y=scaled_deep_var_amounts_95[:display_months],
        mode='lines+markers',
        name='Deep VaR 95%',
        line=dict(color='blue'),
        marker=dict(size=8)
    ))
    
    var_fig.add_trace(go.Scatter(
        x=display_month_strings,
        y=scaled_deep_var_amounts_99[:display_months],
        mode='lines+markers',
        name='Deep VaR 99%',
        line=dict(color='purple'),
        marker=dict(size=8)
    ))
    
    # Add a horizontal line at y=0 to better visualize profit vs loss
    var_fig.add_shape(
        type="line",
        x0=display_month_strings[0],
        y0=0,
        x1=display_month_strings[-1],
        y1=0,
        line=dict(color="black", width=1, dash="dot")
    )
    
    # Update VaR figure layout
    var_fig.update_layout(
        title=dict(text=f'Value at Risk and Expected PnL for ${investment} Investment',
                   x=0.5,  # Center the title horizontally
                   xanchor='center'),
        xaxis_title='Month',
        yaxis_title='Amount ($)',
        legend_title='Legend',
        height=400,
        template='plotly_white',
        hovermode='x unified'
    )
    
     # Create comparison table with both 95% and 99% VaR
    var_comparison = pd.DataFrame({
        'Month': future_month_strings[:display_months],
        'Expected PnL ($)': pnl_values,
        'HS VaR 95% (%)': [model_data['historical_var_95']] * display_months,
        'HS VaR 95% ($)': [historical_var_amount_95] * display_months,
        'HS VaR 99% (%)': [model_data['historical_var_99']] * display_months,
        'HS VaR 99% ($)': [historical_var_amount_99] * display_months,
        'Deep VaR 95% (%)': model_data['deep_var_values_95'][:display_months],
        'Deep VaR 95% ($)': scaled_deep_var_amounts_95[:display_months],
        'Deep VaR 99% (%)': model_data['deep_var_values_99'][:display_months],
        'Deep VaR 99% ($)': scaled_deep_var_amounts_99[:display_months]
        })
    
    table_fig = go.Figure(data=[go.Table(
        header=dict(values=list(var_comparison.columns),
                    # fill_color='paleturquoise',
                    align='left'),
        cells=dict(values=[var_comparison[col] for col in var_comparison.columns],
                #   fill_color='lavender',
                  align='left',
                  format=[None, '.4f', '.2f', '.4f', '.2f', '.2f'])
    )])
    
    table_fig.update_layout(
        title=dict(text=f'VaR and PnL Comparison ({horizon})',
                   x=0.5,
                   xanchor='center'),
        height=400
    )
    
    return fig, future_fig, var_fig, table_fig

# Main app layout
st.markdown(f"### LSTM Model Analysis for {selected_wine_lstm}")
st.markdown("""
This page provides forecasting and risk analysis using Long Short-Term Memory (LSTM) neural networks.
The model predicts future price movements and calculates DeepVaR (Value at Risk) based on the uncertainty in predictions.
""")

# Display plots one by one
try:
    fig, future_fig, var_fig, table_fig = plot_wine_forecast_and_var(
        selected_wine_lstm, forecast_horizon_lstm, investment_lstm
    )
    
    if fig and future_fig and var_fig and table_fig:
        st.plotly_chart(fig, use_container_width=True)
        st.plotly_chart(future_fig, use_container_width=True)
        st.plotly_chart(var_fig, use_container_width=True)
        st.plotly_chart(table_fig, use_container_width=True)
            
except Exception as e:
    st.error(f"Error generating plots: {e}")
    import traceback
    st.error(traceback.format_exc())
