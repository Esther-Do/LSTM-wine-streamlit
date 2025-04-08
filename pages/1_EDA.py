# pages/1_EDA.py
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.seasonal import seasonal_decompose
import numpy as np
import pandas as pd

st.set_page_config(page_title="Exploratory Data Analysis", layout="wide")

st.title(" Exploratory Data Analysis (EDA)")

# --- Access data from session state ---
if 'df_original' not in st.session_state or 'df_returns' not in st.session_state or 'wine_columns' not in st.session_state:
    st.error("Data not loaded. Please go back to the main page.")
    st.stop()

df = st.session_state['df_original']
returns = st.session_state['df_returns']
wine_columns = st.session_state['wine_columns']

# --- EDA Options ---
st.sidebar.header("EDA Options")
selected_wine_eda = st.sidebar.selectbox(
    "Select Fine Wine for Analysis",
    wine_columns,
    key="eda_wine_select"
)

eda_plot_type = st.sidebar.radio(
    "Select Plot Type",
    ('Original Price', 'Log Price', 'Monthly Returns', 'Time Series Decomposition', 'Returns Distribution Windows'),
    key="eda_plot_type"
)

# --- Plotting Functions ---

# 1. Original Price Plot
def plot_original_price(wine_name):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df[wine_name],
        mode='lines',
        name=wine_name
    ))
    fig.update_layout(
        title=f'Original Price Trend for {wine_name}',
        xaxis_title='Date',
        yaxis_title='Price',
        template='plotly_white',
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)

# 2. Log Price Plot
def plot_log_price(wine_name):
    prices = df[wine_name].replace(0, np.nan)  # Replace zeros with NaN
    log_prices = np.log(prices).dropna() # Drop NaN resulting from log(<=0) or original NaNs

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=log_prices.index,
        y=log_prices,
        mode='lines',
        name=f'Log Price of {wine_name}'
    ))
    fig.update_layout(
        title=f'Monthly Log Prices for {wine_name}',
        xaxis_title='Date',
        yaxis_title='Log Price',
        template='plotly_white',
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)

# 3. Returns Plot
def plot_returns(wine_name):
    wine_returns = returns[wine_name].dropna()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=wine_returns.index,
        y=wine_returns,
        mode='lines',
        name=f'Returns of {wine_name}'
    ))
    fig.add_hline(y=0, line=dict(color="black", width=1, dash="dash"))
    fig.update_layout(
        title=f'Monthly Returns for {wine_name}',
        xaxis_title='Date',
        yaxis_title='Returns',
        template='plotly_white',
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)

# 4. Time Series Decomposition Plot
# Cache the decomposition results
@st.cache_data
def get_decomposition(returns_series, wine_name):
    try:
        # Ensure the series has a frequency, default to Monthly Start 'MS' if needed
        if returns_series.index.freq is None:
            returns_series = returns_series.asfreq('MS', method='ffill') # Or choose appropriate frequency/fill method

        decomposition = seasonal_decompose(returns_series.dropna(), model='additive', period=12) # Assuming monthly data, period=12
        return decomposition
    except Exception as e:
        st.error(f"Could not decompose {wine_name}: {e}. Series length might be too short or requires frequency.")
        return None

def plot_decomposition(wine_name):
    time_series = returns[wine_name]
    decomposition = get_decomposition(time_series, wine_name)

    if decomposition:
        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            subplot_titles=(
                f'Original Time Series',
                f'Trend Component',
                f'Seasonal Component',
                f'Residual Component'
            ),
            vertical_spacing=0.08 # Adjust spacing
        )

        dates = decomposition.observed.index # Use index from decomposition results

        fig.add_trace(go.Scatter(x=dates, y=decomposition.observed, mode='lines', name='Original'), row=1, col=1)
        fig.add_trace(go.Scatter(x=dates, y=decomposition.trend, mode='lines', name='Trend'), row=2, col=1)
        fig.add_trace(go.Scatter(x=dates, y=decomposition.seasonal, mode='lines', name='Seasonal'), row=3, col=1)
        fig.add_trace(go.Scatter(x=dates, y=decomposition.resid, mode='lines', name='Residual'), row=4, col=1)

        fig.update_layout(
            height=700,
            title_text=f"{wine_name} - Time Series Decomposition",
            showlegend=False,
            template='plotly_white'
        )
        # Update y-axis titles if desired
        fig.update_yaxes(title_text="Returns", row=1, col=1)
        fig.update_yaxes(title_text="Trend", row=2, col=1)
        fig.update_yaxes(title_text="Seasonality", row=3, col=1)
        fig.update_yaxes(title_text="Residuals", row=4, col=1)

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning(f"Decomposition plot not available for {wine_name}.")


# 5. Returns Distribution Windows Plot
def plot_returns_windows(wine_name):
    wine_returns = returns[wine_name].dropna()
    n_returns = len(wine_returns)

    if n_returns < 30: # Need at least some data for meaningful periods
        st.warning("Not enough data points (< 30) to show distribution windows.")
        return

    # Define periods (e.g., roughly thirds, or fixed lengths like 30)
    period_length = max(30, n_returns // 3) # Use at least 30 points or a third
    period_length = min(period_length, n_returns) # Cap at total length


    fig = make_subplots(rows=3, cols=1,
                        shared_xaxes=True, # Share x-axis for better comparison
                        subplot_titles=('Recent Period', 'Mid Period', 'Early Period'),
                        vertical_spacing=0.1)

    plots_added = 0

    # Recent period
    if n_returns >= period_length:
        history1 = wine_returns[-period_length:]
        percentile_5_1 = np.percentile(history1, 5)
        percentile_1_1 = np.percentile(history1, 1)
        fig.add_trace(go.Histogram(x=history1, name='Recent', nbinsx=50, marker_color='#636EFA'), row=1, col=1)
        fig.add_vline(x=percentile_5_1, line=dict(color="orange", width=2), annotation_text="5%", annotation_position="top right", row=1, col=1)
        fig.add_vline(x=percentile_1_1, line=dict(color="red", width=2), annotation_text="1%", annotation_position="top right", row=1, col=1)
        plots_added += 1

    # Mid period
    if n_returns >= 2 * period_length:
        start_idx_2 = n_returns - 2 * period_length
        end_idx_2 = n_returns - period_length
        history2 = wine_returns[start_idx_2:end_idx_2]
        percentile_5_2 = np.percentile(history2, 5)
        percentile_1_2 = np.percentile(history2, 1)
        fig.add_trace(go.Histogram(x=history2, name='Mid', nbinsx=50, marker_color='#EF553B'), row=2, col=1)
        fig.add_vline(x=percentile_5_2, line=dict(color="orange", width=2), annotation_text="5%", annotation_position="top right", row=2, col=1)
        fig.add_vline(x=percentile_1_2, line=dict(color="red", width=2), annotation_text="1%", annotation_position="top right", row=2, col=1)
        fig.update_yaxes(title_text="Frequency", row=2, col=1)
        plots_added += 1

    # Early period
    if n_returns >= 3 * period_length:
        start_idx_3 = n_returns - 3 * period_length
        end_idx_3 = n_returns - 2 * period_length
        history3 = wine_returns[start_idx_3:end_idx_3]
        percentile_5_3 = np.percentile(history3, 5)
        percentile_1_3 = np.percentile(history3, 1)
        fig.add_trace(go.Histogram(x=history3, name='Early', nbinsx=50, marker_color='#00CC96'), row=3, col=1)
        fig.add_vline(x=percentile_5_3, line=dict(color="orange", width=2), annotation_text="5%", annotation_position="top right", row=3, col=1)
        fig.add_vline(x=percentile_1_3, line=dict(color="red", width=2), annotation_text="1%", annotation_position="top right", row=3, col=1)
        fig.update_xaxes(title_text="Returns", row=3, col=1)
        plots_added += 1

    if plots_added > 0:
        fig.update_layout(
            title_text=f"{wine_name} Returns Distribution at Different Historical Windows",
            height=700,
            showlegend=False,
             template='plotly_white'
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning(f"Not enough distinct periods to display distribution windows for {wine_name}.")


# --- Display Selected Plot ---
st.header(f"Analysis for: {selected_wine_eda}")
st.subheader(eda_plot_type)

if eda_plot_type == 'Original Price':
    plot_original_price(selected_wine_eda)
elif eda_plot_type == 'Log Price':
    plot_log_price(selected_wine_eda)
elif eda_plot_type == 'Monthly Returns':
    plot_returns(selected_wine_eda)
elif eda_plot_type == 'Time Series Decomposition':
    plot_decomposition(selected_wine_eda)
elif eda_plot_type == 'Returns Distribution Windows':
    plot_returns_windows(selected_wine_eda)