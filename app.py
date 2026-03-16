"""
Stock Prediction Dashboard - Streamlit App

Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np


def main():
    """Main Streamlit application."""
    
    # Page config
    st.set_page_config(
        page_title="Stock Prediction Dashboard",
        page_icon="📈",
        layout="wide"
    )
    
    # Title
    st.title("📈 Stock Prediction Dashboard")
    st.markdown("Real-time stock market prediction using machine learning")
    
    # Sidebar
    st.sidebar.title("Configuration")
    
    # Input fields
    symbol = st.sidebar.text_input("Stock Symbol", value="AAPL")
    model_type = st.sidebar.selectbox(
        "Model Type",
        ["ensemble", "arima", "lstm", "gru"],
        index=0
    )
    predict_days = st.sidebar.slider("Days to Predict", 1, 30, 7)
    years = st.sidebar.slider("Years of Data", 1, 10, 5)
    
    # Load button
    if st.sidebar.button("Load Data & Train"):
        with st.spinner("Loading data and training model..."):
            try:
                # Import here to avoid import errors
                from src.models.predictor import StockPredictor
                
                # Create predictor
                predictor = StockPredictor(symbol, model_type)
                
                # Load and train
                predictor.load_data(years=years)
                predictor.train()
                
                # Store in session state
                st.session_state['predictor'] = predictor
                st.session_state['loaded'] = True
                
                st.success("Model trained successfully!")
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.session_state['loaded'] = False
    
    # Check if model is loaded
    if st.session_state.get('loaded', False):
        predictor = st.session_state['predictor']
        
        # Current price
        current_price = predictor.get_current_price()
        
        # Metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Current Price", f"${current_price:.2f}" if current_price else "N/A")
            
        # Make predictions
        predictions = predictor.predict(predict_days)
        
        next_price = predictions['predictions'][0] if len(predictions['predictions']) > 0 else 0
        change = next_price - current_price if current_price else 0
        change_pct = (change / current_price * 100) if current_price else 0
        
        with col2:
            st.metric("Next Day Prediction", f"${next_price:.2f}", 
                     f"{change:+.2f} ({change_pct:+.2f}%)")
                     
        with col3:
            if 'lower_bound' in predictions:
                lower = predictions['lower_bound'][0]
                upper = predictions['upper_bound'][0]
                st.metric("Prediction Range", f"${upper:.2f}", 
                         f"${lower:.2f} - ${upper:.2f}")
        
        # Price chart
        st.subheader("📊 Price History & Predictions")
        
        if predictor.data is not None:
            import plotly.graph_objects as go
            
            fig = go.Figure()
            
            # Price line
            fig.add_trace(go.Scatter(
                x=predictor.data.index,
                y=predictor.data['close'],
                mode='lines',
                name='Close Price',
                line=dict(color='blue', width=2)
            ))
            
            # SMA if available
            if 'sma_20' in predictor.data.columns:
                fig.add_trace(go.Scatter(
                    x=predictor.data.index,
                    y=predictor.data['sma_20'],
                    mode='lines',
                    name='SMA 20',
                    line=dict(color='orange')
                ))
                
            # Predictions
            pred_days = len(predictions['predictions'])
            last_date = predictor.data.index[-1]
            pred_dates = pd.date_range(start=last_date, periods=pred_days + 1, freq='D')[1:]
            
            fig.add_trace(go.Scatter(
                x=pred_dates,
                y=predictions['predictions'],
                mode='lines',
                name='Predictions',
                line=dict(color='red', dash='dash')
            ))
            
            fig.update_layout(
                template='plotly_dark',
                height=500,
                xaxis_title='Date',
                yaxis_title='Price ($)'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        # Predictions table
        st.subheader("📋 Predictions")
        
        pred_df = pd.DataFrame({
            'Day': range(1, predict_days + 1),
            'Predicted Price': predictions['predictions']
        })
        
        if 'lower_bound' in predictions:
            pred_df['Lower Bound'] = predictions['lower_bound']
            pred_df['Upper Bound'] = predictions['upper_bound']
            
        st.dataframe(pred_df, use_container_width=True)
        
        # Technical indicators
        st.subheader("📉 Technical Indicators")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if predictor.data is not None and 'rsi_14' in predictor.data.columns:
                rsi = predictor.data['rsi_14'].iloc[-1]
                st.metric("RSI (14)", f"{rsi:.2f}")
                if rsi > 70:
                    st.warning("⚠️ Overbought")
                elif rsi < 30:
                    st.success("✅ Oversold")
                    
        with col2:
            if predictor.data is not None and 'macd' in predictor.data.columns:
                macd = predictor.data['macd'].iloc[-1]
                signal = predictor.data['macd_signal'].iloc[-1]
                st.metric("MACD", f"{macd:.2f}", f"Signal: {signal:.2f}")
                
    else:
        # Welcome message
        st.info("👈 Please configure your settings in the sidebar and click 'Load Data & Train' to get started!")
        
        # Show sample data
        st.markdown("""
        ### Features:
        - **Data Collection**: Historical stock data from Yahoo Finance
        - **Technical Indicators**: SMA, EMA, RSI, MACD, Bollinger Bands
        - **Machine Learning Models**: ARIMA, LSTM, GRU, Ensemble
        - **Sentiment Analysis**: VADER, TextBlob
        - **Visualization**: Interactive charts with Plotly
        """)


if __name__ == "__main__":
    # Initialize session state
    if 'loaded' not in st.session_state:
        st.session_state['loaded'] = False
    if 'predictor' not in st.session_state:
        st.session_state['predictor'] = None
        
    main()
