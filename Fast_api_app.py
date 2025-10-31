import os
import logging
from datetime import datetime
from typing import List
import json

import torch
from fastapi import FastAPI, Request, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

try:
    from werkzeug.utils import secure_filename
except Exception:
    # Fallback simple sanitizer
    def secure_filename(name: str) -> str:
        return os.path.basename(name)

from scripts.EfficientNet import EfficientNetRegression
from scripts.lstm_model import SolarLSTMForecasting, HybridCNNLSTM
from scripts.preprocess import IRImageProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Solar Forecasting API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")


class SolarForecastingAPI:
    """API class for solar irradiance nowcasting and forecasting"""

    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")

        self.image_processor = IRImageProcessor(target_size=(240, 320))

        self.nowcast_model = None
        self.forecast_model = None
        self.hybrid_model = None

        self.load_models()

    def load_models(self):
        try:
            if os.path.exists('models/best_efficientnet_model.pth'):
                logger.info("Loading CNN nowcasting model...")
                self.nowcast_model = EfficientNetRegression().to(self.device)
                checkpoint = torch.load('models/best_efficientnet_model.pth', map_location=self.device )
                self.nowcast_model.load_state_dict(checkpoint.get("model_state_dict", {}), strict=False)
                self.nowcast_model.eval()
                logger.info("CNN model loaded successfully")

            lstm_ckpt_path = None
            for candidate in ['models/best_lstm_model.pth']:
                if os.path.exists(candidate):
                    lstm_ckpt_path = candidate
                    break

            if lstm_ckpt_path:
                logger.info("Loading LSTM forecasting model...")
                checkpoint = torch.load(lstm_ckpt_path, map_location=self.device, weights_only=False)
                config = checkpoint.get('config', {})

                state_dict_keys = checkpoint.get('model_state_dict', {}).keys()
                inferred_bidirectional = any('lstm.weight_ih_l0_reverse' in k for k in state_dict_keys) if state_dict_keys else True

                use_legacy_head = any(k.startswith('fc.') for k in state_dict_keys) if state_dict_keys else False

                self.forecast_model = SolarLSTMForecasting(
                    input_size=1,
                    hidden_size=config.get('lstm_hidden_size', 128),
                    num_layers=config.get('lstm_num_layers', 2),
                    output_size=config.get('forecast_horizon', 4),
                    bidirectional=config.get('bidirectional', inferred_bidirectional),
                    use_legacy_head=use_legacy_head
                ).to(self.device)

                self.forecast_model.load_state_dict(checkpoint.get('model_state_dict', {}), strict=False)
                self.forecast_model.eval()
                logger.info("LSTM model loaded successfully")

            hybrid_ckpt_path = None
            for candidate in ['models/best_hybrid_model.pth', 'models1/best_hybrid_model.pth']:
                if os.path.exists(candidate):
                    hybrid_ckpt_path = candidate
                    break

            if hybrid_ckpt_path:
                logger.info("Loading hybrid CNN-LSTM model...")
                checkpoint = torch.load(hybrid_ckpt_path, map_location=self.device, weights_only=False)
                config = checkpoint.get('config', {})

                self.hybrid_model = HybridCNNLSTM(
                    sequence_length=config.get('sequence_length', 20),
                    lstm_hidden_size=config.get('lstm_hidden_size', 128),
                    forecast_horizon=config.get('forecast_horizon', 4)
                ).to(self.device)

                self.hybrid_model.load_state_dict(checkpoint.get('model_state_dict', {}), strict=False)
                self.hybrid_model.eval()
                logger.info("Hybrid model loaded successfully")

        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")

    def preprocess_image(self, image_path: str) -> torch.Tensor:
        processed_img = self.image_processor.process_single_image(image_path)
        img_tensor = torch.tensor(processed_img).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0)
        return img_tensor.to(self.device)

    def nowcast_single_image(self, image_path: str):
        if self.nowcast_model is None:
            raise RuntimeError("CNN nowcasting model not loaded")

        img_tensor = self.preprocess_image(image_path)
        with torch.no_grad():
            prediction = self.nowcast_model(img_tensor)
            irradiance = float(prediction.squeeze().cpu().item())

        return {
            'nowcast_irradiance': round(irradiance, 2),
            'timestamp': datetime.now().isoformat(),
            'model': 'CNN_Regression'
        }

    def forecast_from_sequence(self, irradiance_sequence: List[float]):
        if self.forecast_model is None:
            raise RuntimeError("LSTM forecasting model not loaded")

        seq_tensor = torch.tensor(irradiance_sequence, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(self.device)
        with torch.no_grad():
            forecast = self.forecast_model(seq_tensor)
            forecast_values = forecast.squeeze().cpu().numpy().tolist()

        return {
            'forecast_irradiance': [round(float(val), 2) for val in forecast_values],
            'forecast_horizon': len(forecast_values),
            'timestamp': datetime.now().isoformat(),
            'model': 'LSTM_Forecasting'
        }

    def hybrid_predict(self, image_paths: List[str]):
        if self.hybrid_model is None:
            raise RuntimeError("Hybrid CNN-LSTM model not loaded")

        image_tensors = []
        for img_path in image_paths:
            img_tensor = self.preprocess_image(img_path)
            image_tensors.append(img_tensor.squeeze(0))

        sequence_tensor = torch.stack(image_tensors).unsqueeze(0)
        with torch.no_grad():
            nowcasts, forecasts = self.hybrid_model(sequence_tensor)
            nowcast_values = nowcasts.squeeze().cpu().numpy().tolist()
            forecast_values = forecasts.squeeze().cpu().numpy().tolist()

        return {
            'nowcast_sequence': [round(float(val), 2) for val in nowcast_values],
            'forecast_irradiance': [round(float(val), 2) for val in forecast_values],
            'sequence_length': len(nowcast_values),
            'forecast_horizon': len(forecast_values),
            'timestamp': datetime.now().isoformat(),
            'model': 'Hybrid_CNN_LSTM'
        }


api = SolarForecastingAPI()


async def _save_upload_file(upload_file: UploadFile, dest_path: str):
    contents = await upload_file.read()
    with open(dest_path, 'wb') as fh:
        fh.write(contents)
    await upload_file.close()


@app.get('/')
async def index(request: Request):
    return templates.TemplateResponse('index.html', {"request": request})


@app.post('/predict')
async def predict_single(file: UploadFile = File(...)):
    if not file or file.filename == '':
        raise HTTPException(status_code=400, detail="No file uploaded")

    filename = secure_filename(file.filename)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{filename}"
    filepath = os.path.join(UPLOAD_FOLDER, filename)

    try:
        await _save_upload_file(file, filepath)
        result = api.nowcast_single_image(filepath)
        if os.path.exists(filepath):
            os.remove(filepath)
        return JSONResponse(result)
    except Exception as e:
        if os.path.exists(filepath):
            os.remove(filepath)
        logger.error(f"Error in single prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class ForecastRequest(BaseModel):
    irradiance_sequence: List[float]


@app.post('/forecast')
async def forecast_sequence(req: ForecastRequest):
    sequence = req.irradiance_sequence
    if not isinstance(sequence, list) or len(sequence) < 10:
        raise HTTPException(status_code=400, detail='Sequence must be a list with at least 10 values')

    try:
        result = api.forecast_from_sequence(sequence)
        return JSONResponse(result)
    except Exception as e:
        logger.error(f"Error in forecasting: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post('/hybrid_predict')
async def hybrid_predict(files: List[UploadFile] = File(...)):
    if not files or len(files) < 10:
        raise HTTPException(status_code=400, detail='At least 10 images required for sequence prediction')

    saved_paths: List[str] = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    try:
        for i, file in enumerate(files):
            if file.filename:
                filename = secure_filename(file.filename)
                filename = f"{timestamp}_{i:03d}_{filename}"
                filepath = os.path.join(UPLOAD_FOLDER, filename)
                await _save_upload_file(file, filepath)
                saved_paths.append(filepath)

        result = api.hybrid_predict(saved_paths)

        for p in saved_paths:
            if os.path.exists(p):
                os.remove(p)

        return JSONResponse(result)

    except Exception as e:
        for p in saved_paths:
            if os.path.exists(p):
                os.remove(p)
        logger.error(f"Error in hybrid prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get('/health')
async def health_check():
    model_status = {
        'cnn_loaded': api.nowcast_model is not None,
        'lstm_loaded': api.forecast_model is not None,
        'hybrid_loaded': api.hybrid_model is not None
    }

    return JSONResponse({
        'status': 'healthy',
        'models': model_status,
        'device': str(api.device),
        'timestamp': datetime.now().isoformat()
    })


@app.get('/api/training-data')
async def training_data():
    try:
        logs_dir = 'logs'
        result = {}
        if not os.path.isdir(logs_dir):
            return JSONResponse(result)

        for fname in os.listdir(logs_dir):
            if fname.endswith('.json'):
                path = os.path.join(logs_dir, fname)
                try:
                    with open(path, 'r', encoding='utf-8') as fh:
                        result[fname] = json.load(fh)
                except Exception as e:
                    logger.warning(f"Could not read {path}: {e}")
                    result[fname] = None

        return JSONResponse(result)

    except Exception as e:
        logger.error(f"Error reading training data logs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == '__main__':
    import uvicorn
    # Bind to 127.0.0.1 so you can open the site in a browser at http://127.0.0.1:5000
    # Using 0.0.0.0 is fine for server binding, but most browsers cannot open that
    # address directly (ERR_ADDRESS_INVALID). Run with localhost/127.0.0.1 instead.
    uvicorn.run(app, host='127.0.0.1', port=5000, reload=True)
