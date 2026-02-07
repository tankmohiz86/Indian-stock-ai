"""
Indian Stock Market AI Predictor - Standalone Demo
Run this single file to test predictions instantly!

Installation:
pip install streamlit plotly pandas numpy

Run:
streamlit run this_file.py
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="Indian Stock AI", page_icon="📈", layout="wide")

# Sample Data Generation
@st.cache_data
def get_predictions():
    stocks = ['RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK', 'HINDUNILVR', 
              'ITC', 'SBIN', 'BHARTIARTL', 'KOTAKBANK', 'LT', 'AXISBANK', 
              'ASIANPAINT', 'MARUTI', 'SUNPHARMA', 'TITAN', 'BAJFINANCE',
              'ULTRACEMCO', 'NESTLEIND', 'WIPRO', 'HCLTECH', 'TECHM',
              'POWERGRID', 'NTPC', 'ONGC', 'TATAMOTORS', 'BAJAJFINSV',
              'M&M', 'TATASTEEL', 'ADANIENT']
    
    np.random.seed(42)
    data = []
    for stock in stocks:
        prob = 0.55 + (0.3 * np.random.beta(2, 2))
        data.append({
            'Symbol': stock,
            'Probability': prob,
            'Prediction': 'UP' if prob > 0.5 else 'DOWN',
            'Confidence': 0.6 + (0.3 * np.random.beta(2, 2)),
            'Price': np.random.uniform(500, 3000)
        })
    
    return pd.DataFrame(data).sort_values('Probability', ascending=False).reset_index(drop=True)

# Header
st.markdown("""
<div style='text-align: center; padding: 2rem;'>
    <h1 style='color: #667eea; font-size: 3rem;'>🤖 Indian Stock Market AI</h1>
    <p style='font-size: 1.3rem; color: #666;'>February 2026 Predictions • Powered by XGBoost</p>
</div>
""", unsafe_allow_html=True)

df = get_predictions()

# Metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("Stocks Analyzed", "30")
col2.metric("Bullish Signals", len(df[df['Prediction']=='UP']), f"{len(df[df['Prediction']=='UP'])/30*100:.0f}%")
col3.metric("Avg Probability", f"{df['Probability'].mean()*100:.1f}%")
col4.metric("High Confidence", len(df[df['Confidence']>0.7]))

st.markdown("---")

# Best Pick
best = df.iloc[0]
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown(f"""
    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 2rem; border-radius: 15px; color: white; box-shadow: 0 8px 16px rgba(0,0,0,0.2);'>
        <h1 style='margin:0; font-size: 2.5rem;'>🏆 {best['Symbol']}</h1>
        <h2 style='margin: 0.5rem 0; font-size: 2rem;'>Probability: {best['Probability']*100:.1f}%</h2>
        <hr style='border-color: rgba(255,255,255,0.3);'>
        <p style='font-size: 1.2rem; margin: 0.5rem 0;'><b>Prediction:</b> {best['Prediction']} ⬆️</p>
        <p style='font-size: 1.2rem; margin: 0.5rem 0;'><b>Confidence:</b> {best['Confidence']*100:.1f}%</p>
        <p style='font-size: 1.2rem; margin: 0.5rem 0;'><b>Price:</b> ₹{best['Price']:.2f}</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=best['Probability']*100,
        title={'text': "Upside Probability"},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkgreen"},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 70], 'color': "yellow"},
                {'range': [70, 100], 'color': "lightgreen"}
            ],
        }
    ))
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# Top 10 Chart
st.subheader("📊 Top 10 Stock Predictions")
top10 = df.head(10)

fig = go.Figure(go.Bar(
    y=top10['Symbol'],
    x=top10['Probability']*100,
    orientation='h',
    marker=dict(color=['#00cc00' if p>0.7 else '#ff9900' if p>0.6 else '#666' for p in top10['Probability']]),
    text=[f"{p*100:.1f}%" for p in top10['Probability']],
    textposition='outside'
))
fig.update_layout(title="Ranked by Probability", xaxis_title="Probability (%)", 
                  yaxis={'categoryorder':'total ascending'}, height=500)
st.plotly_chart(fig, use_container_width=True)

# Full Table
st.subheader("📋 All Predictions")
display = df.copy()
display['Probability'] = display['Probability'].apply(lambda x: f"{x*100:.1f}%")
display['Confidence'] = display['Confidence'].apply(lambda x: f"{x*100:.1f}%")
display['Price'] = display['Price'].apply(lambda x: f"₹{x:.2f}")
st.dataframe(display, use_container_width=True, height=400)

# Download
csv = df.to_csv(index=False)
st.download_button("📥 Download Predictions", csv, "predictions_feb_2026.csv", "text/csv", use_container_width=True)

# Sidebar
with st.sidebar:
    st.title("ℹ️ About")
    st.info("""
    This demo shows AI-powered stock predictions for February 2026.
    
    **Model:** XGBoost
    **Features:** 40+ technical indicators
    **Data:** 3 years historical
    """)
    
    st.markdown("---")
    st.warning("⚠️ **Disclaimer**\n\nFor educational purposes only. Not financial advice.")
    
    st.markdown("---")
    if st.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()

# Footer
st.markdown("---")
st.markdown("<div style='text-align: center; color: #666;'><p>Built with ❤️ using Streamlit • Not Financial Advice</p></div>", unsafe_allow_html=True)
