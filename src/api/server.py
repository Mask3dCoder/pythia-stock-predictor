"""
Prediction API Server

FastAPI-based REST API for stock predictions.
"""

import logging
import os
from typing import Optional, Dict
from datetime import datetime

import uvicorn
from fastapi import FastAPI, HTTPException, Header, Depends
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Check if FastAPI is available
try:
    from fastapi import FastAPI
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    logger.warning("FastAPI not available. Install fastapi and uvicorn.")


class PredictionRequest(BaseModel):
    """Request model for predictions."""
    symbol: str
    model_type: str = "ensemble"
    days: int = 7


class PredictionResponse(BaseModel):
    """Response model for predictions."""
    symbol: str
    predictions: list
    lower_bound: Optional[list] = None
    upper_bound: Optional[list] = None
    current_price: Optional[float] = None
    timestamp: str


# SECURITY: Add API key authentication
async def verify_api_key(x_api_key: Optional[str] = Header(None)):
    """Verify API key from header."""
    expected_key = os.environ.get('API_KEY', 'default_dev_key')
    if x_api_key != expected_key:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return x_api_key


class PredictionAPI:
    """FastAPI-based prediction server."""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize Prediction API.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.predictors = {}
        
        if not FASTAPI_AVAILABLE:
            raise ImportError("FastAPI is required. Install with: pip install fastapi uvicorn")
            
        # Create FastAPI app
        self.app = FastAPI(
            title="Stock Prediction API",
            description="Real-time stock market prediction API",
            version="1.0.0"
        )
        
        # Setup routes
        self._setup_routes()
        
    def _setup_routes(self):
        """Setup API routes."""
        
        @self.app.get("/")
        def root():
            """Root endpoint."""
            return {
                "name": "Stock Prediction API",
                "version": "1.0.0",
                "status": "running"
            }
            
        @self.app.get("/health")
        def health():
            """Health check endpoint."""
            return {"status": "healthy"}
            
        @self.app.post("/predict", response_model=PredictionResponse)
        async def predict(request: PredictionRequest, api_key: str = Depends(verify_api_key)):
            """Make stock predictions.
            
            NOTE: For production, consider adding rate limiting using slowapi:
            from slowapi import Limiter
            from slowapi.util import get_remote_address
            """
            try:
                # Get or create predictor
                predictor = self._get_predictor(request.symbol, request.model_type)
                
                # Make predictions
                predictions = predictor.predict(request.days)
                
                # Get current price
                current_price = predictor.get_current_price()
                
                return PredictionResponse(
                    symbol=request.symbol,
                    predictions=predictions['predictions'].tolist(),
                    lower_bound=predictions.get('lower_bound', predictions['predictions']).tolist(),
                    upper_bound=predictions.get('upper_bound', predictions['predictions']).tolist(),
                    current_price=current_price,
                    timestamp=datetime.now().isoformat()
                )
                
            except Exception as e:
                logger.error(f"Prediction error: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))
                
        @self.app.get("/predict/{symbol}")
        def predict_symbol(
            symbol: str,
            model_type: str = "ensemble",
            days: int = 7
        ):
            """Make predictions for a symbol (GET request)."""
            try:
                predictor = self._get_predictor(symbol, model_type)
                predictions = predictor.predict(days)
                current_price = predictor.get_current_price()
                
                return {
                    "symbol": symbol,
                    "predictions": predictions['predictions'].tolist(),
                    "lower_bound": predictions.get('lower_bound', predictions['predictions']).tolist(),
                    "upper_bound": predictions.get('upper_bound', predictions['predictions']).tolist(),
                    "current_price": current_price,
                    "timestamp": datetime.now().isoformat()
                }
                
            except Exception as e:
                logger.error(f"Prediction error: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))
                
        @self.app.get("/stock/{symbol}")
        def get_stock_info(symbol: str):
            """Get stock information."""
            try:
                predictor = self._get_predictor(symbol, "ensemble")
                
                # Get stock info
                info = predictor.collector.get_stock_info(symbol)
                
                # Get current data
                data = predictor.data
                
                return {
                    "symbol": symbol,
                    "info": info,
                    "current_price": predictor.get_current_price(),
                    "data_points": len(data) if data is not None else 0,
                    "timestamp": datetime.now().isoformat()
                }
                
            except Exception as e:
                logger.error(f"Error getting stock info: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))
                
        @self.app.get("/models")
        def list_models():
            """List available models."""
            return {
                "models": ["arima", "lstm", "gru", "ensemble"]
            }
            
    def _get_predictor(self, symbol: str, model_type: str):
        """Get or create predictor for symbol."""
        key = f"{symbol}_{model_type}"
        
        if key not in self.predictors:
            from src.models.predictor import StockPredictor
            self.predictors[key] = StockPredictor(
                symbol=symbol,
                model_type=model_type,
                config=self.config
            )
            # Load data and train
            self.predictors[key].load_data(years=5)
            self.predictors[key].train()
            
        return self.predictors[key]
        
    def run(self, host: str = "0.0.0.0", port: int = 8000):
        """
        Run the API server.
        
        Args:
            host: Host to bind to
            port: Port to listen on
        """
        logger.info(f"Starting API server on {host}:{port}")
        uvicorn.run(self.app, host=host, port=port)


def create_app(config: Optional[Dict] = None) -> FastAPI:
    """
    Create FastAPI application.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        FastAPI app
    """
    api = PredictionAPI(config)
    return api.app


def run_server(host: str = "0.0.0.0", port: int = 8000):
    """
    Run the prediction API server.
    
    Args:
        host: Host to bind to
        port: Port to listen on
    """
    api = PredictionAPI()
    api.run(host, port)
