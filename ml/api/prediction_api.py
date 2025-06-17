"""
FastAPI web service for currency price predictions.

This module provides REST API endpoints for making price predictions
using trained ML models.
"""

import sys
import os
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime
import asyncio
from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import json

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from utils.model_inference import ModelPredictor, PredictionResult
from config.training_config import MLConfig
from utils.logging_utils import MLLogger

# Global predictor instance
predictor = None
app_logger = None


class PredictionRequest(BaseModel):
    """Request model for single prediction."""
    currency_pair: str = Field(..., description="Currency pair to predict (e.g., 'Divine Orb -> Chaos Orb')")
    prediction_horizon_days: int = Field(1, ge=1, le=30, description="Days ahead to predict (1-30)")


class MultiplePredictionRequest(BaseModel):
    """Request model for multiple predictions."""
    currency_pairs: Optional[List[str]] = Field(None, description="List of currency pairs (None for all)")
    prediction_horizon_days: int = Field(1, ge=1, le=30, description="Days ahead to predict (1-30)")
    top_n: Optional[int] = Field(None, ge=1, le=100, description="Limit to top N results")
    sort_by: str = Field("price_change_percent", description="Field to sort by")
    ascending: bool = Field(False, description="Sort order")


class PredictionResponse(BaseModel):
    """Response model for predictions."""
    success: bool
    timestamp: str
    predictions: List[Dict[str, Any]]
    metadata: Dict[str, Any]


class ModelStatusResponse(BaseModel):
    """Response model for model status."""
    loaded_models: int
    available_models: Dict[str, Dict[str, Any]]
    last_update: str
    status: str


# FastAPI app initialization
app = FastAPI(
    title="PoEconomy Price Prediction API",
    description="REST API for Path of Exile currency price predictions",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Initialize the predictor on startup."""
    global predictor, app_logger
    
    app_logger = MLLogger("PredictionAPI")
    app_logger.info("Starting Prediction API...")
    
    try:
        # Find the latest model directory
        models_base = Path("models")
        if models_base.exists():
            currency_dirs = [d for d in models_base.iterdir() 
                           if d.is_dir() and d.name.startswith('currency_')]
            if currency_dirs:
                latest_dir = max(currency_dirs, key=lambda d: d.stat().st_mtime)
                
                # Initialize predictor
                config = MLConfig()
                predictor = ModelPredictor(latest_dir, config, app_logger)
                
                # Load models
                available_models = predictor.load_available_models()
                app_logger.info(f"Loaded {len(available_models)} models successfully")
            else:
                app_logger.error("No model directories found")
        else:
            app_logger.error("Models directory not found")
            
    except Exception as e:
        app_logger.error(f"Failed to initialize predictor: {str(e)}")


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information."""
    return {
        "service": "PoEconomy Price Prediction API",
        "version": "1.0.0",
        "status": "running",
        "timestamp": datetime.now().isoformat(),
        "docs": "/docs"
    }


@app.get("/health", response_model=Dict[str, str])
async def health_check():
    """Health check endpoint."""
    global predictor
    
    status = "healthy" if predictor is not None else "unhealthy"
    return {
        "status": status,
        "timestamp": datetime.now().isoformat(),
        "models_loaded": len(predictor.loaded_models) if predictor else 0
    }


@app.get("/models/status", response_model=ModelStatusResponse)
async def get_model_status():
    """Get status of loaded models."""
    global predictor
    
    if not predictor:
        raise HTTPException(status_code=503, detail="Predictor not initialized")
    
    try:
        available_models = predictor.load_available_models()
        
        return ModelStatusResponse(
            loaded_models=len(available_models),
            available_models=available_models,
            last_update=datetime.now().isoformat(),
            status="operational"
        )
    except Exception as e:
        app_logger.error(f"Failed to get model status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models/currencies", response_model=List[str])
async def get_available_currencies():
    """Get list of available currency pairs."""
    global predictor
    
    if not predictor:
        raise HTTPException(status_code=503, detail="Predictor not initialized")
    
    try:
        return list(predictor.loaded_models.keys())
    except Exception as e:
        app_logger.error(f"Failed to get currencies: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict", response_model=PredictionResponse)
async def predict_single(request: PredictionRequest):
    """Make a prediction for a single currency pair."""
    global predictor
    
    if not predictor:
        raise HTTPException(status_code=503, detail="Predictor not initialized")
    
    try:
        result = predictor.predict_price(
            request.currency_pair,
            request.prediction_horizon_days
        )
        
        if not result:
            raise HTTPException(
                status_code=404, 
                detail=f"Unable to generate prediction for {request.currency_pair}"
            )
        
        return PredictionResponse(
            success=True,
            timestamp=datetime.now().isoformat(),
            predictions=[result.to_dict()],
            metadata={
                "model_type": result.model_type,
                "features_used": result.features_used,
                "data_points_used": result.data_points_used
            }
        )
        
    except Exception as e:
        app_logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/multiple", response_model=PredictionResponse)
async def predict_multiple(request: MultiplePredictionRequest):
    """Make predictions for multiple currency pairs."""
    global predictor
    
    if not predictor:
        raise HTTPException(status_code=503, detail="Predictor not initialized")
    
    try:
        if request.top_n:
            # Get top predictions
            results = predictor.get_top_predictions(
                top_n=request.top_n,
                sort_by=request.sort_by,
                ascending=request.ascending
            )
        else:
            # Get specific currencies or all
            results = predictor.predict_multiple_currencies(
                request.currency_pairs,
                request.prediction_horizon_days
            )
        
        if not results:
            raise HTTPException(
                status_code=404,
                detail="No predictions could be generated"
            )
        
        # Calculate summary metadata
        avg_change = sum(r.price_change_percent for r in results) / len(results)
        avg_confidence = sum(r.confidence_score for r in results) / len(results)
        positive_predictions = sum(1 for r in results if r.price_change_percent > 0)
        
        return PredictionResponse(
            success=True,
            timestamp=datetime.now().isoformat(),
            predictions=[r.to_dict() for r in results],
            metadata={
                "total_predictions": len(results),
                "average_change_percent": round(avg_change, 2),
                "average_confidence": round(avg_confidence, 3),
                "positive_predictions": positive_predictions,
                "positive_percentage": round(positive_predictions / len(results) * 100, 1)
            }
        )
        
    except Exception as e:
        app_logger.error(f"Multiple prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/predict/top", response_model=PredictionResponse)
async def get_top_predictions(
    top_n: int = Query(10, ge=1, le=100, description="Number of top predictions"),
    sort_by: str = Query("price_change_percent", description="Field to sort by"),
    ascending: bool = Query(False, description="Sort order"),
    horizon: int = Query(1, ge=1, le=30, description="Prediction horizon in days")
):
    """Get top N predictions."""
    global predictor
    
    if not predictor:
        raise HTTPException(status_code=503, detail="Predictor not initialized")
    
    try:
        # Temporarily set prediction horizon
        results = []
        for currency_pair in predictor.loaded_models.keys():
            result = predictor.predict_price(currency_pair, horizon)
            if result:
                results.append(result)
        
        if not results:
            raise HTTPException(status_code=404, detail="No predictions available")
        
        # Sort results
        try:
            sorted_results = sorted(
                results,
                key=lambda x: getattr(x, sort_by),
                reverse=not ascending
            )[:top_n]
        except AttributeError:
            sorted_results = results[:top_n]
        
        # Calculate metadata
        avg_change = sum(r.price_change_percent for r in sorted_results) / len(sorted_results)
        avg_confidence = sum(r.confidence_score for r in sorted_results) / len(sorted_results)
        positive = sum(1 for r in sorted_results if r.price_change_percent > 0)
        
        return PredictionResponse(
            success=True,
            timestamp=datetime.now().isoformat(),
            predictions=[r.to_dict() for r in sorted_results],
            metadata={
                "total_predictions": len(sorted_results),
                "sorted_by": sort_by,
                "ascending": ascending,
                "prediction_horizon_days": horizon,
                "average_change_percent": round(avg_change, 2),
                "average_confidence": round(avg_confidence, 3),
                "positive_predictions": positive,
                "positive_percentage": round(positive / len(sorted_results) * 100, 1)
            }
        )
        
    except Exception as e:
        app_logger.error(f"Top predictions failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/data/current-league", response_model=Dict[str, Any])
async def get_current_league_info():
    """Get information about the current league."""
    global predictor
    
    if not predictor:
        raise HTTPException(status_code=503, detail="Predictor not initialized")
    
    try:
        current_data = predictor.get_current_league_data(days_back=1)
        
        if current_data is None or current_data.empty:
            raise HTTPException(status_code=404, detail="No current league data available")
        
        # Get league information
        league_info = current_data.iloc[0]
        
        return {
            "league_name": league_info.get('league_name', 'Unknown'),
            "league_start": league_info.get('league_start', 'Unknown'),
            "league_day": int(league_info.get('league_day', 0)),
            "currency_pairs_available": int(current_data['currency_pair'].nunique()),
            "total_data_points": len(current_data),
            "last_update": current_data['date'].max().isoformat(),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        app_logger.error(f"Failed to get league info: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/models/reload")
async def reload_models(background_tasks: BackgroundTasks):
    """Reload models in the background."""
    global predictor
    
    if not predictor:
        raise HTTPException(status_code=503, detail="Predictor not initialized")
    
    def reload_task():
        try:
            app_logger.info("Reloading models...")
            available_models = predictor.load_available_models()
            app_logger.info(f"Reloaded {len(available_models)} models")
        except Exception as e:
            app_logger.error(f"Model reload failed: {str(e)}")
    
    background_tasks.add_task(reload_task)
    
    return {
        "message": "Model reload initiated",
        "timestamp": datetime.now().isoformat()
    }


if __name__ == "__main__":
    import uvicorn
    
    # Run the API server
    uvicorn.run(
        "prediction_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 