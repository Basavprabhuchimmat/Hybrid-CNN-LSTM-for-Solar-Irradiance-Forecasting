import os
from typing import List
import torch
import torch.nn.functional as F
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import logging
from datetime import datetime
import aiofiles
from pathlib import Path

# Import our models
from cnn_model import SolarCNNRegression, SolarCNNWithFeatureExtraction
from lstm_model import SolarLSTMForecasting
from preprocess import IRImageProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Solar Forecasting API",
    description="API for solar irradiance nowcasting and forecasting using CNN and LSTM models",
    version="1.0.0"
)

# Configure templates and static files
templates = Jinja2Templates(directory="templates")

# Request/Response models
class ForecastRequest(BaseModel):
    irradiance_sequence: List[float]

class NowcastResponse(BaseModel):
    nowcast_irradiance: float
    timestamp: str
    model: str

class ForecastResponse(BaseModel):
    forecast_irradiance: List[float]
    forecast_horizon: int
    timestamp: str
    model: str

class HealthResponse(BaseModel):
    status: str
    models: dict
    device: str
    timestamp: str

# Configuration
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB
UPLOAD_FOLDER = "uploads"

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

class SolarForecastingAPI:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")

        # Initialize image processor
        self.image_processor = IRImageProcessor(target_size=(240, 320))

        # Model placeholders
        self.nowcast_model = None
        self.forecast_model = None

        # Load models
        self.load_models()

    def load_models(self):
        try:
            # Load CNN nowcasting model
            if os.path.exists('models/best_cnn_model.pth'):
                logger.info("Loading CNN nowcasting model...")
                self.nowcast_model = SolarCNNRegression().to(self.device)
                checkpoint = torch.load('models/best_cnn_model.pth', map_location=self.device)
                self.nowcast_model.load_state_dict(checkpoint['model_state_dict'])
                self.nowcast_model.eval()
                logger.info("CNN model loaded successfully")

            # Load LSTM forecasting model
            if os.path.exists('models/best_lstm_model.pth'):
                logger.info("Loading LSTM forecasting model...")
                checkpoint = torch.load('models/best_lstm_model.pth', map_location=self.device, weights_only=False)
                config = checkpoint.get('config', {})
                self.forecast_model = SolarLSTMForecasting(
                    input_size=1,
                    hidden_size=config.get('lstm_hidden_size', 128),
                    num_layers=config.get('lstm_num_layers', 2),
                    output_size=config.get('forecast_horizon', 4)
                ).to(self.device)
                self.forecast_model.load_state_dict(checkpoint['model_state_dict'])
                self.forecast_model.eval()
                logger.info("LSTM model loaded successfully")

        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")

    def preprocess_image(self, image_path):
        try:
            # Process the image
            processed_img = self.image_processor.process_single_image(image_path)
            # Convert to tensor
            img_tensor = torch.tensor(processed_img).permute(2, 0, 1).float() / 255.0
            img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension
            return img_tensor.to(self.device)
        except Exception as e:
            logger.error(f"Error preprocessing image: {str(e)}")
            raise

    async def nowcast_single_image(self, image_path):
        if self.nowcast_model is None:
            raise ValueError("CNN nowcasting model not loaded")

        try:
            # Preprocess image
            img_tensor = self.preprocess_image(image_path)

            # Predict
            with torch.no_grad():
                prediction = self.nowcast_model(img_tensor)
                irradiance = prediction.item()

            return {
                'nowcast_irradiance': round(irradiance, 2),
                'timestamp': datetime.now().isoformat(),
                'model': 'CNN_Regression'
            }

        except Exception as e:
            logger.error(f"Error in nowcasting: {str(e)}")
            raise

    async def forecast_from_sequence(self, irradiance_sequence):
        if self.forecast_model is None:
            raise ValueError("LSTM forecasting model not loaded")

        try:
            # Convert to tensor
            seq_tensor = torch.tensor(irradiance_sequence, dtype=torch.float32)
            seq_tensor = seq_tensor.unsqueeze(0).unsqueeze(-1)  # (1, seq_len, 1)
            seq_tensor = seq_tensor.to(self.device)

            # Predict
            with torch.no_grad():
                forecast = self.forecast_model(seq_tensor)
                forecast_values = forecast.squeeze().cpu().numpy().tolist()

            return {
                'forecast_irradiance': [round(val, 2) for val in forecast_values],
                'forecast_horizon': len(forecast_values),
                'timestamp': datetime.now().isoformat(),
                'model': 'LSTM_Forecasting'
            }

        except Exception as e:
            logger.error(f"Error in forecasting: {str(e)}")
            raise

# Initialize API
api_instance = SolarForecastingAPI()

# Routes
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Serve the main page"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_model=NowcastResponse)
async def predict_single(file: UploadFile = File(...)):
    """
    Upload an infrared image and get solar irradiance nowcast prediction
    """
    try:
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file selected")

        # Check file size
        contents = await file.read()
        if len(contents) > MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail="File too large")

        # Save uploaded file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{file.filename}"
        filepath = os.path.join(UPLOAD_FOLDER, filename)

        # Write file asynchronously
        async with aiofiles.open(filepath, 'wb') as f:
            await f.write(contents)

        try:
            # Perform nowcasting
            result = await api_instance.nowcast_single_image(filepath)

            # Clean up
            os.remove(filepath)

            return result

        except Exception as e:
            # Clean up on error
            if os.path.exists(filepath):
                os.remove(filepath)
            raise HTTPException(status_code=500, detail=str(e))

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in single prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/forecast", response_model=ForecastResponse)
async def forecast_sequence(request: ForecastRequest):
    """
    Provide a sequence of irradiance values and get future forecasts
    """
    try:
        sequence = request.irradiance_sequence

        if not isinstance(sequence, list) or len(sequence) < 10:
            raise HTTPException(
                status_code=400, 
                detail="Sequence must be a list with at least 10 values"
            )

        # Perform forecasting
        result = await api_instance.forecast_from_sequence(sequence)
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in forecasting: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Check the health status of the API and loaded models
    """
    model_status = {
        'cnn_loaded': api_instance.nowcast_model is not None,
        'lstm_loaded': api_instance.forecast_model is not None
    }

    return {
        'status': 'healthy',
        'models': model_status,
        'device': str(api_instance.device),
        'timestamp': datetime.now().isoformat()
    }

# Exception handlers
@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    return JSONResponse(
        status_code=404,
        content={"error": "Endpoint not found"}
    )

@app.exception_handler(500)
async def internal_error_handler(request: Request, exc):
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error"}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=5000,
        reload=True,
        log_level="info"
    )
