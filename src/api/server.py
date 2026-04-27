"""
Prediction API Server

FastAPI-based REST API for stock predictions.

Enhanced with:
- Rate limiting
- Improved security
- WebSocket support for real-time updates
- Better error handling
"""

import logging
import os
from typing import Optional, Dict
from datetime import datetime
from collections import defaultdict
from threading import Lock

import uvicorn
from fastapi import FastAPI, HTTPException, Header, Depends, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator

from src.core.exceptions import APIError, format_exception

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
    symbol: str = Field(..., min_length=1, max_length=10)
    model_type: str = Field(default="ensemble", pattern="^(arima|lstm|gru|ensemble)$")
    days: int = Field(default=7, ge=1, le=365)


class PredictionResponse(BaseModel):
    """Response model for predictions."""
    symbol: str
    predictions: list
    lower_bound: Optional[list] = None
    upper_bound: Optional[list] = None
    current_price: Optional[float] = None
    timestamp: str


class RateLimiter:
    """Simple in-memory rate limiter."""
    
    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        self.requests = defaultdict(list)
        self.lock = Lock()
    
    def is_allowed(self, client_id: str) -> bool:
        """Check if request is allowed for client."""
        now = datetime.now()
        minute_ago = now.timestamp() - 60
        
        with self.lock:
            # Clean old requests
            self.requests[client_id] = [
                ts for ts in self.requests[client_id] if ts > minute_ago
            ]
            
            # Check limit
            if len(self.requests[client_id]) >= self.requests_per_minute:
                return False
            
            # Add current request
            self.requests[client_id].append(now.timestamp())
            return True


# Rate limiter instance
rate_limiter = RateLimiter(requests_per_minute=60)


def get_client_id(request: Request) -> str:
    """Get client identifier from request."""
    return request.client.host if request.client else "unknown"


async def verify_api_key(x_api_key: Optional[str] = Header(None)):
    """Verify API key from header. Only enforced if API_KEY env var is set."""
    expected_key = os.environ.get('API_KEY')

    # If no API key is configured, skip auth (development / unconfigured mode)
    if not expected_key:
        return None

    if x_api_key != expected_key:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return x_api_key


async def check_rate_limit(request: Request, api_key: str = None):
    """Check rate limit for client using API key as identifier."""
    client_id = api_key if api_key else get_client_id(request)
    if not rate_limiter.is_allowed(client_id):
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Please try again later."
        )


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
            version="3.0.0"
        )
        
        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Configure appropriately for production
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
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
                "version": "3.0.0",
                "status": "running"
            }
            
        @self.app.get("/health")
        def health():
            """Health check endpoint."""
            return {"status": "healthy"}
        
        @self.app.websocket("/ws/predict")
        async def websocket_predict(websocket: WebSocket):
            """WebSocket endpoint for real-time predictions."""
            expected_key = os.environ.get('API_KEY')
            if expected_key:
                client_key = websocket.headers.get('x-api-key') or websocket.query_params.get('api_key')
                if client_key != expected_key:
                    await websocket.close(code=4001, reason="Invalid API key")
                    return

            await websocket.accept()
            try:
                while True:
                    # Receive message
                    data = await websocket.receive_json()
                    symbol = data.get("symbol")
                    model_type = data.get("model_type", "ensemble")
                    days = data.get("days", 7)
                    
                    if not symbol:
                        await websocket.send_json({"error": "Symbol is required"})
                        continue
                    
                    # Get or create predictor
                    predictor = self._get_predictor(symbol, model_type)
                    
                    # Make predictions
                    predictions = predictor.predict(days)
                    current_price = predictor.get_current_price()
                    
                    # Send response
                    await websocket.send_json({
                        "symbol": symbol,
                        "predictions": predictions['predictions'].tolist(),
                        "lower_bound": predictions.get('lower_bound', predictions['predictions']).tolist(),
                        "upper_bound": predictions.get('upper_bound', predictions['predictions']).tolist(),
                        "current_price": current_price,
                        "timestamp": datetime.now().isoformat()
                    })
                    
            except WebSocketDisconnect:
                logger.info("WebSocket disconnected")
            except Exception as e:
                logger.error(f"WebSocket error: {str(e)}")
                await websocket.send_json({"error": str(e)})
            
        @self.app.post("/predict", response_model=PredictionResponse)
        async def predict(
            request: PredictionRequest,
            req: Request,
            api_key: str = Depends(verify_api_key)
        ):
            """Make stock predictions with rate limiting."""
            # Check rate limit
            await check_rate_limit(req, api_key)
            
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
        async def predict_symbol(
            symbol: str,
            req: Request,
            model_type: str = "ensemble",
            days: int = 7,
            api_key: str = Depends(verify_api_key)
        ):
            """Make predictions for a symbol (GET request)."""
            await check_rate_limit(req, api_key)
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
        async def get_stock_info(
            symbol: str,
            req: Request,
            api_key: str = Depends(verify_api_key)
        ):
            """Get stock information."""
            await check_rate_limit(req, api_key)
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
        def list_models(api_key: str = Depends(verify_api_key)):
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


app = create_app()
