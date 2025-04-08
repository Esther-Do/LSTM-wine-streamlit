# pages/2_HS_VaR.py
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd

st.set_page_config(page_title="Historical Simulation VaR", layout="wide")

st.title("Historical Simulation Value at Risk (HS VaR)")

# --- Access data from session state ---
if 'df_returns' not in st.session_state or 'wine_columns' not in st.session_state:
    st.error("Data not loaded. Please go back to the main page.")
    st.stop()

returns = st.session_state['df_returns']
wine_columns = st.session_state['wine_columns']

# --- User Inputs ---
st.sidebar.header("HS VaR Options")
selected_wine_var = st.sidebar.selectbox(
    "Select Wine",
    wine_columns,
    key="var_wine_select"
)

investment_var = st.sidebar.number_input(
    "Investment Amount ($)",
    min_value=100,
    value=10000, # Default value matching notebook more closely
    step=100,
    key="var_investment"
)

confidence_level_var = st.sidebar.slider(
    "Confidence Level (%)",
    min_value=90,
    max_value=99,
    value=95, # Default value
    step=1,
    key="var_confidence"
)

# --- VaR Calculation Function ---
def hs_var(returns_series, alpha=95, time_frame='month'):
    """Calculates Historical VaR for monthly data"""
    if time_frame == 'month':
        time_factor = 1
    elif time_frame == 'year':
        time_factor = np.sqrt(12) # Scale volatility for annual VaR
    else:
        raise ValueError("'time_frame' must be 'month' or 'year'")

    q = 100 - alpha # Calculate percentile (e.g., 5 for 95% confidence)
    # VaR is the negative of the q-th percentile of the returns distribution
    var_percent = -np.percentile(returns_series.dropna(), q)
    # Scale by time factor
    var_scaled_percent = var_percent * time_factor
    return np.round(var_scaled_percent, 4)

# --- Calculate and Display Results ---
st.header(f"VaR Analysis for: {selected_wine_var}")

wine_returns_var = returns[selected_wine_var].dropna()

if len(wine_returns_var) < 20: # Need sufficient historical data
     st.warning(f"Not enough historical return data ({len(wine_returns_var)} points) for reliable VaR calculation for {selected_wine_var}.")
else:
    # Calculate VaR
    varM_percent = hs_var(wine_returns_var, alpha=confidence_level_var, time_frame='month')
    varY_percent = hs_var(wine_returns_var, alpha=confidence_level_var, time_frame='year')

    varM_amount = investment_var * varM_percent
    varY_amount = investment_var * varY_percent

    # Display VaR Summary Table
    st.subheader(f"{confidence_level_var}% Value at Risk Summary")
    var_data = {
        'Time Horizon': ['Monthly', 'Annual'],
        f'VaR ({confidence_level_var}%)': [f"{varM_percent:.2%}", f"{varY_percent:.2%}"],
        f'VaR Amount ($)': [f"${varM_amount:,.2f}", f"${varY_amount:,.2f}"]
    }
    st.dataframe(pd.DataFrame(var_data))

    st.markdown(f"""
    *Interpretation:* With {confidence_level_var}% confidence, the maximum expected loss for a ${investment_var:,.2f}
    investment in '{selected_wine_var}' over the next month is approximately **${varM_amount:,.2f}**.
    Over the next year, it is approximately **${varY_amount:,.2f}**, assuming return characteristics remain similar.
    """)

    # Plot Returns Distribution with VaR line
    st.subheader("Returns Distribution and VaR Threshold")
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(
        x=wine_returns_var,
        nbinsx=50,
        name='Returns Distribution',
        marker_color='blue',
        opacity=0.7
    ))

    # Add VaR line (use the monthly VaR percentage threshold)
    var_threshold = -varM_percent # VaR percentage is negative loss, threshold is the return value
    fig_hist.add_vline(
        x=var_threshold,
        line=dict(color="red", width=2, dash="dash"),
        annotation_text=f"{confidence_level_var}% Monthly VaR Threshold ({var_threshold:.2%})",
        annotation_position="top left",
    )

    # Add 1% and 5% percentile lines for context
    p5 = np.percentile(wine_returns_var, 5)
    p1 = np.percentile(wine_returns_var, 1)

    fig_hist.add_vline(x=p5, line=dict(color="orange", width=1, dash="dot"), annotation_text=f"5th Pctl ({p5:.2%})", annotation_position="bottom left")
    fig_hist.add_vline(x=p1, line=dict(color="purple", width=1, dash="dot"), annotation_text=f"1st Pctl ({p1:.2%})", annotation_position="bottom right")


    fig_hist.update_layout(
        title=f'Monthly Returns Distribution for {selected_wine_var}',
        xaxis_title='Monthly Return',
        yaxis_title='Frequency',
        template='plotly_white',
        height=500
    )
    st.plotly_chart(fig_hist, use_container_width=True)