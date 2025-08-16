import os
import torch
import torch.nn.functional as F
import cv2
import numpy as np
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import logging
from datetime import datetime

# Import our models
from cnn_model import SolarCNNRegression, SolarCNNWithFeatureExtraction
from lstm_model import SolarLSTMForecasting
from preprocess import IRImageProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

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

    def nowcast_single_image(self, image_path):
        
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

    def forecast_from_sequence(self, irradiance_sequence):
        
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
api = SolarForecastingAPI()

@app.route('/')
def index():
    
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_single():
    
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        if file:
            # Save uploaded file
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{timestamp}_{filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            try:
                # Perform nowcasting
                result = api.nowcast_single_image(filepath)

                # Clean up
                os.remove(filepath)

                return jsonify(result)

            except Exception as e:
                # Clean up on error
                if os.path.exists(filepath):
                    os.remove(filepath)
                raise

    except Exception as e:
        logger.error(f"Error in single prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/forecast', methods=['POST'])
def forecast_sequence():
    
    try:
        data = request.get_json()

        if 'irradiance_sequence' not in data:
            return jsonify({'error': 'No irradiance sequence provided'}), 400

        sequence = data['irradiance_sequence']

        if not isinstance(sequence, list) or len(sequence) < 10:
            return jsonify({'error': 'Sequence must be a list with at least 10 values'}), 400

        # Perform forecasting
        result = api.forecast_from_sequence(sequence)

        return jsonify(result)

    except Exception as e:
        logger.error(f"Error in forecasting: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():

    model_status = {
        'cnn_loaded': api.nowcast_model is not None,
        'lstm_loaded': api.forecast_model is not None
    }

    return jsonify({
        'status': 'healthy',
        'models': model_status,
        'device': str(api.device),
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
