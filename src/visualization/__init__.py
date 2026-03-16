"""
Visualization Module for Stock Prediction CLI

This module handles dashboard creation and plotting using Plotly and Streamlit.
"""

from .dashboard import StockDashboard
from .plots import StockPlotter

__all__ = ['StockDashboard', 'StockPlotter']
