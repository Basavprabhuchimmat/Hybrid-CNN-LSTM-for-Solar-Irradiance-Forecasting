# ☀️ Hybrid CNN-LSTM for Solar Irradiance Forecasting 🌤️

[![Python](https://img.shields.io/badge/Python-3.9-blue?logo=python)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12-orange?logo=tensorflow)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Last Update](https://img.shields.io/badge/Last%20Update-July%202025-brightgreen)]()

> 🚀 A deep learning-based approach for short-term solar irradiance forecasting using a hybrid CNN-LSTM model with infrared image processing, inspired by cutting-edge research.

---

## 📌 Project Overview

Accurate solar irradiance forecasting is critical for renewable energy planning and grid stability. This project implements a **Hybrid CNN-LSTM model** that combines **Convolutional Neural Networks (CNNs)** for spatial feature extraction and **Long Short-Term Memory (LSTM)** networks for temporal prediction, leveraging **infrared satellite imagery** and **NASA POWER** datasets.

---

## 🧠 Methodology

### ✅ Steps Involved:
1. **Data Collection**
   - Infrared satellite images
   - Global Solar Irradiance (GSI) data from NASA POWER

2. **Preprocessing**
   - Image resizing & normalization
   - Time series formatting for LSTM

3. **Model Architecture**
   - **CNN** layers for spatial feature extraction
   - **LSTM** layers for capturing temporal dependencies
   - Fully connected layers for prediction

4. **Forecasting**
   - Output: Short-term solar irradiance (GSI) predictions

---

## 🗂️ Project Structure

Hybrid-CNN-LSTM-for-Solar-Irradiance-Forecasting/
│
├── data/                       # Preprocessed dataset & images
├── models/                     # Saved model weights and architecture
├── notebook/                   # Jupyter notebooks for training & evaluation
├── app/                        # Flask web app interface (if implemented)
├── utils/                      # Helper scripts for preprocessing, visualization
├── README.md                   # Project documentation
└── requirements.txt            # Dependencies


## 🌐 Web Interface (Optional)

A simple Flask-based dashboard allows users to:
- Upload infrared satellite images
- Get predicted solar irradiance
- Visualize time series graphs

---

## 📊 Results

- 📈 **Model Accuracy**: ~XX% (to be updated with metrics)
- 🧪 Evaluated using MAE, RMSE, and R² Score
- 🖼️ Visualizations for actual vs predicted irradiance over time

<p align="center">
  <img src="notebook/output_plot.png" width="600"/>
</p>

---

## 🧰 Tech Stack

- Python 3.9+
- TensorFlow / Keras
- NumPy, Pandas
- OpenCV
- Matplotlib & Seaborn
- NASA POWER API
- Flask (for Web UI)



## ⚙️ Installation

```bash
# Clone the repo
git clone https://github.com/Anand-b-patil/Hybrid-CNN-LSTM-for-Solar-Irradiance-Forecasting.git
cd Hybrid-CNN-LSTM-for-Solar-Irradiance-Forecasting

# Set up a virtual environment
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
````

---

## 🚀 Run the Model

```bash
# Step 1: Preprocess the data
python utils/preprocess.py

# Step 2: Train the model
python notebook/train_model.py

# Step 3: Run prediction
python notebook/predict.py
```

---

## 📸 Sample Output

| Input Image (IR)                    | Predicted GSI                         |
| ----------------------------------- | ------------------------------------- |
| ![input](notebook/sample_input.png) | ![output](notebook/sample_output.png) |

---

## 📚 References

* Paper: [A Hybrid CNN-LSTM Framework and Infrared Image Processing for Solar Irradiance Forecasting](https://doi.org/10.xxxx/xxx)
* [NASA POWER Dataset](https://power.larc.nasa.gov/)
* [Keras Documentation](https://keras.io/)

---

## 🤝 Contributing

Contributions, ideas, and suggestions are welcome! Feel free to fork the repo and submit a pull request.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 🙋‍♂️ Author

**Anand Bhimagouda Patil**
📧 [anand.b.patil@example.com](mailto:anand.b.patil@example.com)
🔗 [LinkedIn](https://www.linkedin.com/in/your-profile) | [GitHub](https://github.com/Anand-b-patil)

---
