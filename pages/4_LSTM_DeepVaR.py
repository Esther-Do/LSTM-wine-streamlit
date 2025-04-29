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
    
    # Calculate confidence intervals
    ci_lower, ci_upper = calculate_95ci(y_pred_lstm_inverse)
    
    # Calculate RMSE
    rmse = root_mean_squared_error(y_test_inverse, y_pred_lstm_inverse)
    
    # Generate future predictions (12 months)
    future_predictions = []
    # Get the last lag values from the time series
    last_sequence = time_series[-lag:].reshape(1, lag, 1)
    
    # Make future predictions one step at a time
    for _ in range(12):
        # Predict the next value
        next_pred = model.predict(last_sequence, verbose=0)
        # Store the prediction
        future_predictions.append(next_pred[0, 0])
        # Update the sequence by removing the first value and adding the new prediction at the end
        # Create a new sequence with the updated values
        last_sequence = np.roll(last_sequence, -1, axis=1)
        last_sequence[0, -1, 0] = next_pred[0, 0]
    
    # Convert predictions to numpy array and reshape
    future_predictions = np.array(future_predictions).reshape(-1, 1)
    # Inverse transform to get actual values
    future_predictions_inverse = scaler.inverse_transform(future_predictions)
    
    # Calculate confidence intervals for future predictions
    future_lower, future_upper = calculate_95ci(future_predictions_inverse)
    
    # Calculate Historical VaR (95%)
    historical_var = hs_var_calc(returns[wine_name])
    
    # Calculate DeepVaR using future predictions
    # For simplicity, we'll use the lower bound of the confidence interval
    # as a proxy for VaR (this is a simplified approach)
    deep_var_values = []
    deep_var_amounts = []
    
    for i in range(len(future_predictions_inverse)):
        # Calculate potential loss as difference between prediction and lower bound
        potential_loss = (future_predictions_inverse[i] - future_lower[i]) / future_predictions_inverse[i]
        deep_var_values.append(float(potential_loss))
        
        # Calculate VaR amount for $1000 investment
        deep_var_amounts.append(float(potential_loss * 1000))
    
    return {
        'time_series': time_series,
        'lag': lag,
        'train_size': train_size,
        'y_test_inverse': y_test_inverse,
        'y_pred_lstm_inverse': y_pred_lstm_inverse,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'rmse': rmse,
        'future_predictions': future_predictions_inverse,
        'future_lower': future_lower,
        'future_upper': future_upper,
        'historical_var': historical_var,
        'deep_var_values': deep_var_values,
        'deep_var_amounts': deep_var_amounts
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
    ci_lower = model_data['ci_lower']
    ci_upper = model_data['ci_upper']
    rmse = model_data['rmse']
    future_predictions = model_data['future_predictions']
    future_lower = model_data['future_lower']
    future_upper = model_data['future_upper']
    historical_var = model_data['historical_var']
    deep_var_values = model_data['deep_var_values']
    deep_var_amounts = model_data['deep_var_amounts']
    
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
        name='Test Predictions',
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
    display_lower = future_lower[:display_months]
    display_upper = future_upper[:display_months]
    
    # Add future predictions line
    future_fig.add_trace(go.Scatter(
        x=display_month_strings, 
        y=display_predictions.flatten(),
        mode='lines+markers',
        name=f'Future Predictions ({horizon})',
        line=dict(color='purple'),
        marker=dict(size=8)
    ))
    
    # Add future confidence interval - upper bound
    future_fig.add_trace(go.Scatter(
        x=display_month_strings,
        y=display_upper.flatten(),
        mode='lines',
        line=dict(width=0),
        showlegend=False
    ))
    
    # Add future confidence interval - lower bound
    future_fig.add_trace(go.Scatter(
        x=display_month_strings,
        y=display_lower.flatten(),
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
    
    # Calculate historical VaR amount for the current investment
    historical_var_amount = investment * historical_var
    
    # Scale Deep VaR amounts for the current investment
    scaled_deep_var_amounts = [amount * (investment / 1000) for amount in deep_var_amounts]
    
    # Calculate Profit/Loss based on future predictions
    # Assuming future_predictions are percentage changes
    pnl_values = [investment * pred for pred in future_predictions[:display_months].flatten()]
    
    # Add Historical VaR line
    var_fig.add_trace(go.Scatter(
        x=display_month_strings,
        y=[historical_var_amount] * len(display_month_strings),
        mode='lines',
        name='HS VaR',
        line=dict(color='red', dash='dash')
    ))
    
    # Add Deep VaR line
    var_fig.add_trace(go.Scatter(
        x=display_month_strings,
        y=scaled_deep_var_amounts[:display_months],
        mode='lines+markers',
        name='Deep VaR',
        line=dict(color='blue'),
        marker=dict(size=8)
    ))
    
    # Add PnL line
    var_fig.add_trace(go.Scatter(
        x=display_month_strings,
        y=pnl_values,
        mode='lines+markers',
        name='Expected PnL',
        line=dict(color='green'),
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
    
    # Create a table to show VaR comparison
    var_comparison = pd.DataFrame({
        'Month': future_month_strings[:display_months],
        'HS VaR (%)': [historical_var] * display_months,
        'HS VaR ($)': [historical_var_amount] * display_months,
        'Deep VaR (%)': deep_var_values[:display_months],
        'Deep VaR ($)': scaled_deep_var_amounts[:display_months],
        'Expected PnL ($)': pnl_values
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
