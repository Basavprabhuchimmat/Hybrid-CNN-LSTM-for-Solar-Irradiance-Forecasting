import os
import torch
import cv2
import numpy as np
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import logging
from datetime import datetime

# Import our models
from scripts.EfficientNet import EfficientNetRegression
from scripts.lstm_model import SolarLSTMForecasting, HybridCNNLSTM
from scripts.preprocess import IRImageProcessor

# ----------------------------
# Logging Configuration
# ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("SolarForecasting")

# ----------------------------
# Flask App Config
# ----------------------------
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB
app.config['UPLOAD_FOLDER'] = os.environ.get("UPLOAD_FOLDER", "uploads")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ----------------------------
# Forecasting API Class
# ----------------------------
class SolarForecastingAPI:
    """Handles model loading and inference for solar irradiance"""

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        self.image_processor = IRImageProcessor(target_size=(240, 320))

        self.nowcast_model = None
        self.forecast_model = None
        self.hybrid_model = None

        self.load_models()

    def load_models(self):
        """Load models from checkpoints"""
        try:
            # ---------------- CNN ----------------
            cnn_path = "models/best_efficientnet_model.pth"
            if os.path.exists(cnn_path):
                logger.info("Loading CNN nowcasting model...")
                self.nowcast_model = EfficientNetRegression().to(self.device)
                checkpoint = torch.load(cnn_path, map_location=self.device)
                self.nowcast_model.load_state_dict(checkpoint["model_state_dict"], strict=False)
                self.nowcast_model.eval()
                logger.info("CNN model loaded successfully")

            # ---------------- LSTM ----------------
            lstm_path = "models/best_lstm_model.pth"
            if os.path.exists(lstm_path):
                logger.info("Loading LSTM forecasting model...")
                checkpoint = torch.load(lstm_path, map_location=self.device)
                config = checkpoint.get("config", {})
                self.forecast_model = SolarLSTMForecasting(
                    input_size=1,
                    hidden_size=config.get("lstm_hidden_size", 128),
                    num_layers=config.get("lstm_num_layers", 2),
                    output_size=config.get("forecast_horizon", 4),
                    bidirectional=config.get("bidirectional", True)
                ).to(self.device)
                self.forecast_model.load_state_dict(checkpoint["model_state_dict"], strict=False)
                self.forecast_model.eval()
                logger.info("LSTM model loaded successfully")

            # ---------------- Hybrid ----------------
            for hybrid_path in ["models/best_hybrid_model.pth", "models1/best_hybrid_model.pth"]:
                if os.path.exists(hybrid_path):
                    logger.info("Loading Hybrid CNN-LSTM model...")
                    checkpoint = torch.load(hybrid_path, map_location=self.device)
                    config = checkpoint.get("config", {})
                    self.hybrid_model = HybridCNNLSTM(
                        sequence_length=config.get("sequence_length", 20),
                        lstm_hidden_size=config.get("lstm_hidden_size", 128),
                        forecast_horizon=config.get("forecast_horizon", 4)
                    ).to(self.device)
                    self.hybrid_model.load_state_dict(checkpoint["model_state_dict"], strict=False)
                    self.hybrid_model.eval()
                    logger.info("Hybrid model loaded successfully")
                    break

        except Exception as e:
            logger.exception(f"Error loading models: {e}")

    def preprocess_image(self, image_path):
        """Preprocess image into tensor"""
        img = self.image_processor.process_single_image(image_path)
        if len(img.shape) == 2:  # grayscale -> RGB
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        tensor = torch.tensor(img).permute(2, 0, 1).float() / 255.0
        return tensor.unsqueeze(0).to(self.device)

    # -------- Inference Methods --------
    def nowcast_single_image(self, image_path):
        if self.nowcast_model is None:
            raise RuntimeError("CNN nowcasting model not loaded")
        img_tensor = self.preprocess_image(image_path)
        with torch.no_grad():
            pred = self.nowcast_model(img_tensor)
        return {
            "nowcast_irradiance": round(pred.item(), 2),
            "timestamp": datetime.now().isoformat(),
            "model": "CNN_Regression"
        }

    def forecast_from_sequence(self, irradiance_sequence):
        if self.forecast_model is None:
            raise RuntimeError("LSTM forecasting model not loaded")
        seq_tensor = torch.tensor(irradiance_sequence, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(self.device)
        with torch.no_grad():
            forecast = self.forecast_model(seq_tensor).squeeze().cpu().numpy().tolist()
        return {
            "forecast_irradiance": [round(x, 2) for x in forecast],
            "forecast_horizon": len(forecast),
            "timestamp": datetime.now().isoformat(),
            "model": "LSTM_Forecasting"
        }

    def hybrid_predict(self, image_paths):
        if self.hybrid_model is None:
            raise RuntimeError("Hybrid CNN-LSTM model not loaded")
        tensors = [self.preprocess_image(p).squeeze(0) for p in image_paths]
        seq_tensor = torch.stack(tensors).unsqueeze(0).to(self.device)
        with torch.no_grad():
            nowcasts, forecasts = self.hybrid_model(seq_tensor)
        return {
            "nowcast_sequence": [round(x, 2) for x in nowcasts.squeeze().cpu().numpy().tolist()],
            "forecast_irradiance": [round(x, 2) for x in forecasts.squeeze().cpu().numpy().tolist()],
            "timestamp": datetime.now().isoformat(),
            "model": "Hybrid_CNN_LSTM"
        }


# ----------------------------
# Routes
# ----------------------------
api = SolarForecastingAPI()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict_single():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "" or not allowed_file(file.filename):
        return jsonify({"error": "Invalid file"}), 400

    filepath = None
    try:
        filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{secure_filename(file.filename)}"
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        result = api.nowcast_single_image(filepath)
        return jsonify(result)
    except Exception as e:
        logger.exception("Error in /predict")
        return jsonify({"error": str(e)}), 500
    finally:
        if filepath and os.path.exists(filepath):
            os.remove(filepath)


@app.route("/forecast", methods=["POST"])
def forecast_sequence():
    data = request.get_json(force=True, silent=True)
    if not data or "irradiance_sequence" not in data:
        return jsonify({"error": "No irradiance sequence provided"}), 400
    sequence = data["irradiance_sequence"]
    if not isinstance(sequence, list) or len(sequence) < 10:
        return jsonify({"error": "Sequence must be list with at least 10 values"}), 400
    try:
        result = api.forecast_from_sequence(sequence)
        return jsonify(result)
    except Exception as e:
        logger.exception("Error in /forecast")
        return jsonify({"error": str(e)}), 500


@app.route("/hybrid_predict", methods=["POST"])
def hybrid_predict():
    if "files" not in request.files:
        return jsonify({"error": "No files uploaded"}), 400
    files = request.files.getlist("files")
    if len(files) < 10:
        return jsonify({"error": "At least 10 images required"}), 400

    filepaths = []
    try:
        for i, file in enumerate(files):
            if file and allowed_file(file.filename):
                fname = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{i:03d}_{secure_filename(file.filename)}"
                fpath = os.path.join(app.config["UPLOAD_FOLDER"], fname)
                file.save(fpath)
                filepaths.append(fpath)

        result = api.hybrid_predict(filepaths)
        return jsonify(result)
    except Exception as e:
        logger.exception("Error in /hybrid_predict")
        return jsonify({"error": str(e)}), 500
    finally:
        for f in filepaths:
            if os.path.exists(f):
                os.remove(f)


@app.route("/health")
def health_check():
    return jsonify({
        "status": "healthy",
        "device": str(api.device),
        "models": {
            "cnn_loaded": api.nowcast_model is not None,
            "lstm_loaded": api.forecast_model is not None,
            "hybrid_loaded": api.hybrid_model is not None
        },
        "timestamp": datetime.now().isoformat()
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("DEBUG", "true").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug)
