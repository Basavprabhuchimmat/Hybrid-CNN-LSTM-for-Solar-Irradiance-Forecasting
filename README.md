☀️ Hybrid CNN-LSTM for Solar Irradiance Forecasting 🌤️
Python TensorFlow License: MIT Last Update

🚀 A deep learning-based approach for short-term solar irradiance forecasting using a hybrid CNN-LSTM model with infrared image processing, inspired by cutting-edge research.

📌 Project Overview
Accurate solar irradiance forecasting is critical for renewable energy planning and grid stability. This project implements a Hybrid CNN-LSTM model that combines Convolutional Neural Networks (CNNs) for spatial feature extraction and Long Short-Term Memory (LSTM) networks for temporal prediction, leveraging infrared satellite imagery and NASA POWER datasets.

🧠 Methodology
✅ Steps Involved:
Data Collection

Infrared satellite images
Global Solar Irradiance (GSI) data from NASA POWER
Preprocessing

Image resizing & normalization
Time series formatting for LSTM
Model Architecture

CNN layers for spatial feature extraction
LSTM layers for capturing temporal dependencies
Fully connected layers for prediction
Forecasting

Output: Short-term solar irradiance (GSI) predictions
🗂️ Project Structure
Hybrid-CNN-LSTM-for-Solar-Irradiance-Forecasting/│

│

├── data/ # Preprocessed dataset & images

├── models/ # Saved model weights and architecture

├── notebook/ # Jupyter notebooks for training & evaluation

├── app/ # Flask web app interface (if implemented)

├── utils/ # Helper scripts for preprocessing, visualization

├── README.md # Project documentation

└── requirements.txt # Dependencies

🌐 Web Interface (Optional)
A simple Flask-based dashboard allows users to:

Upload infrared satellite images
Get predicted solar irradiance
Visualize time series graphs
📊 Results
📈 Model Accuracy: ~XX% (to be updated with metrics)
🧪 Evaluated using MAE, RMSE, and R² Score
🖼️ Visualizations for actual vs predicted irradiance over time


🧰 Tech Stack
Python 3.9+
TensorFlow / Keras
NumPy, Pandas
OpenCV
Matplotlib & Seaborn
NASA POWER API
Flask (for Web UI)
⚙️ Installation
# Clone the repo
git clone https://github.com/Anand-b-patil/Hybrid-CNN-LSTM-for-Solar-Irradiance-Forecasting.git
cd Hybrid-CNN-LSTM-for-Solar-Irradiance-Forecasting

# Set up a virtual environment
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
🚀 Run the Model
# Step 1: Preprocess the data
python utils/preprocess.py

# Step 2: Train the model
python notebook/train_model.py

# Step 3: Run prediction
python notebook/predict.py
📸 Sample Output
Input Image (IR)	Predicted GSI
input	output
📚 References
Paper: A Hybrid CNN-LSTM Framework and Infrared Image Processing for Solar Irradiance Forecasting
NASA POWER Dataset
Keras Documentation
🤝 Contributing
Contributions, ideas, and suggestions are welcome! Feel free to fork the repo and submit a pull request.

📄 License
This project is licensed under the MIT License.

🙋‍♂️ Author
Anand Bhimagouda Patil 📧 anand.b.patil@example.com 🔗 LinkedIn | GitHub

