"""
Stock Dashboard Module

Creates interactive web dashboards using Streamlit.
"""

import logging
import sys
from typing import Optional, Dict
from pathlib import Path

logger = logging.getLogger(__name__)

# Check if streamlit is available
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False
    logger.warning("Streamlit not available. Install streamlit for dashboard functionality.")


class StockDashboard:
    """Streamlit-based stock prediction dashboard."""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize Stock Dashboard.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        if not STREAMLIT_AVAILABLE:
            raise ImportError("Streamlit is required for the dashboard. Install with: pip install streamlit")
            
    def run(self, predictor=None):
        """
        Run the Streamlit dashboard.
        
        Args:
            predictor: StockPredictor instance
        """
        if not STREAMLIT_AVAILABLE:
            print("Streamlit not available. Please install streamlit to run the dashboard.")
            return
            
        import streamlit as st
        import pandas as pd
        import numpy as np
        
        # Set page config
        st.set_page_config(
            page_title="Stock Prediction Dashboard",
            page_icon="📈",
            layout="wide"
        )
        
        # Sidebar
        st.sidebar.title("Stock Prediction")
        
        # Input section
        symbol = st.sidebar.text_input("Stock Symbol", value="AAPL")
        
        model_type = st.sidebar.selectbox(
            "Model Type",
            ["ensemble", "arima", "lstm", "gru"],
            index=0
        )
        
        predict_days = st.sidebar.slider("Days to Predict", 1, 30, 7)
        
        # Main content
        st.title(f"📈 Stock Prediction: {symbol}")
        
        if predictor is None:
            # Create predictor if not provided
            from src.models.predictor import StockPredictor
            
            with st.spinner("Loading data and training model..."):
                try:
                    predictor = StockPredictor(symbol, model_type, self.config)
                    predictor.load_data(years=5)
                    predictor.train()
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    return
        else:
            symbol = predictor.symbol
            
        # Current price section
        col1, col2, col3 = st.columns(3)
        
        current_price = predictor.get_current_price()
        
        with col1:
            st.metric("Current Price", f"${current_price:.2f}" if current_price else "N/A")
            
        # Make predictions
        with st.spinner("Making predictions..."):
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
        
        # Charts section
        st.subheader("📊 Price History")
        
        if predictor.data is not None:
            # Plot price chart
            import plotly.graph_objects as go
            
            fig = go.Figure()
            
            # Add price line
            fig.add_trace(go.Scatter(
                x=predictor.data.index,
                y=predictor.data['close'],
                mode='lines',
                name='Close Price',
                line=dict(color='blue', width=2)
            ))
            
            # Add SMA if available
            if 'sma_20' in predictor.data.columns:
                fig.add_trace(go.Scatter(
                    x=predictor.data.index,
                    y=predictor.data['sma_20'],
                    mode='lines',
                    name='SMA 20',
                    line=dict(color='orange')
                ))
                
            if 'sma_50' in predictor.data.columns:
                fig.add_trace(go.Scatter(
                    x=predictor.data.index,
                    y=predictor.data['sma_50'],
                    mode='lines',
                    name='SMA 50',
                    line=dict(color='green')
                ))
                
            # Add prediction
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
            
            if 'lower_bound' in predictions:
                fig.add_trace(go.Scatter(
                    x=pred_dates,
                    y=predictions['upper_bound'],
                    mode='lines',
                    name='Upper Bound',
                    line=dict(color='gray', width=0),
                    showlegend=False
                ))
                
                fig.add_trace(go.Scatter(
                    x=pred_dates,
                    y=predictions['lower_bound'],
                    mode='lines',
                    name='Confidence Interval',
                    line=dict(color='gray', width=0),
                    fill='tonexty',
                    fillcolor='rgba(255, 0, 0, 0.2)'
                ))
            
            fig.update_layout(
                template='plotly_dark',
                height=500,
                xaxis_title='Date',
                yaxis_title='Price ($)'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        # Technical Indicators
        st.subheader("📉 Technical Indicators")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if 'rsi_14' in predictor.data.columns:
                rsi = predictor.data['rsi_14'].iloc[-1]
                st.metric("RSI (14)", f"{rsi:.2f}")
                if rsi > 70:
                    st.warning("⚠️ Overbought")
                elif rsi < 30:
                    st.success("✅ Oversold")
                    
        with col2:
            if 'macd' in predictor.data.columns and 'macd_signal' in predictor.data.columns:
                macd = predictor.data['macd'].iloc[-1]
                signal = predictor.data['macd_signal'].iloc[-1]
                st.metric("MACD", f"{macd:.2f}", f"Signal: {signal:.2f}")
                
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
        
        # Model evaluation
        st.subheader("📊 Model Evaluation")
        
        try:
            metrics = predictor.evaluate()
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("MAE", f"${metrics['mae']:.2f}")
            with col2:
                st.metric("RMSE", f"${metrics['rmse']:.2f}")
            with col3:
                st.metric("R²", f"{metrics['r2']:.4f}")
            with col4:
                st.metric("MAPE", f"{metrics['mape']:.2f}%")
                
        except Exception as e:
            st.warning(f"Could not evaluate model: {e}")
            
        # Info section
        st.sidebar.info(
            f"""
            **Model Type:** {model_type}
            **Prediction Days:** {predict_days}
            **Data Points:** {len(predictor.data) if predictor.data is not None else 0}
            """
        )


def run_dashboard(predictor=None):
    """Run the Streamlit dashboard."""
    dashboard = StockDashboard()
    dashboard.run(predictor)


if __name__ == "__main__":
    # When run directly with `streamlit run`, execute the dashboard
    import yaml
    
    # Load config
    config = {}
    try:
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
    except Exception:
        pass
    
    run_dashboard()
