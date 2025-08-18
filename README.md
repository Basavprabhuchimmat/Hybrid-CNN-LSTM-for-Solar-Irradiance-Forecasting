

# 🌤️ Hybrid CNN-LSTM for Solar Irradiance Forecasting 

[![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)](https://www.python.org/)  [![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12-orange?logo=tensorflow)](https://www.tensorflow.org/)  [![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)  ![Last Update](https://img.shields.io/badge/Last%20Update-Aug%202025-brightgreen)

> 🚀 A deep learning-based approach for **short-term solar irradiance forecasting** using a **Hybrid CNN-LSTM** model with infrared image processing, inspired by cutting-edge research.

---

## 📌 Project Overview

Accurate solar irradiance forecasting is critical for **renewable energy planning** and **grid stability** ⚡.  

This project implements a **Hybrid CNN-LSTM model** that:
- 🖼️ Uses **Convolutional Neural Networks (CNNs)** for spatial feature extraction  
- ⏳ Uses **Long Short-Term Memory (LSTM)** for temporal prediction  
- 🔬 Leverages **infrared satellite imagery** and **NASA POWER datasets**

---

## 🧠 Methodology

### ✅ Steps Involved
1. **Data Collection**
   - Infrared satellite images 🛰️
   - Global Solar Irradiance (GSI) data from NASA POWER  

2. **Preprocessing**
   - Image resizing & normalization 🖼️
   - Time series formatting for LSTM ⏳  

3. **Model Architecture**
   - CNN layers → spatial feature extraction  
   - LSTM layers → capture temporal dependencies  
   - Fully connected layers → final prediction  

4. **Forecasting**
   - Output → Short-term **solar irradiance (GSI)** predictions ☀️📈  

---

## 🗂️ Project Structure

```

Hybrid-CNN-LSTM-for-Solar-Irradiance-Forecasting/
│
├── data/             # Preprocessed dataset & images
├── models/           # Saved model weights and architecture
├── notebook/         # Jupyter notebooks for training & evaluation
├── app/              # Flask web app interface (if implemented)
├── scripts/            # Helper scripts for preprocessing, visualization
├── README.md         # Project documentation
└── requirements.txt  # Dependencies

````

---

## 🌐 Web Interface (Optional)

A simple **Flask-based dashboard** 🖥️ allows users to:
- Upload infrared satellite images  
- Get predicted solar irradiance  
- Visualize time series graphs 📊  

---

## 📊 Results

- 📈 **Model Accuracy**: ~XX% (update with metrics)  
- 🧪 Evaluated using: **MAE, RMSE, R² Score**  
- 🖼️ Visualization: Actual vs Predicted irradiance over time  

<p align="center">
  <img src="notebook/output_plot.png" width="600" alt="Predicted vs Actual Plot"/>
</p>

---

## 🧰 Tech Stack

- 🐍 Python 3.9+  
- 🧠 TensorFlow / Keras  
- 🔢 NumPy, Pandas  
- 🎨 OpenCV, Matplotlib, Seaborn  
- 🌐 Flask (for Web UI)  

---

## ⚙️ Installation

```bash
# Clone the repo
git clone https://github.com/Anand-b-patil/Hybrid-CNN-LSTM-for-Solar-Irradiance-Forecasting.git
cd Hybrid-CNN-LSTM-for-Solar-Irradiance-Forecasting

# Create a virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
````

---

## 🚀 Run the Model

```bash
# Step 1: Preprocess the data
python scripts/preprocess.py

# Step 2: Train the model
python scripts/train_model.py

# Step 3: Run prediction
python app.py
```

---

## 📸 Sample Output

| Input Image (IR)                    | Predicted GSI                         |
| ----------------------------------- | ------------------------------------- |
| ![input](notebook/sample_input.png) | ![output](notebook/sample_output.png) |

---

## 📚 References

* 📜 [A Hybrid CNN-LSTM Framework and Infrared Image Processing for Solar Irradiance Forecasting](https://ieeexplore.ieee.org/document/10906220)
* 🌍 [GIRASOL Dataset](https://doi.org/10.1016/j.dib.2021.106914)
* 🧠 [Keras Documentation](https://keras.io/)

---

## 🤝 Contributing

Contributions, ideas, and suggestions are welcome! 💡
Feel free to **fork the repo** and submit a **pull request** 🌟

---

## 📄 License

This project is licensed under the **MIT License** 📝

---

## 🙋‍♂️ Author

**Anand Bhimagouda Patil**
📧 [anand.b.patil@example.com](mailto:ap6272440@gmail.com)
🔗 [GitHub](https://github.com/Anand-b-patil) | [LinkedIn](https://linkedin.com/in/anand_b_patil)


