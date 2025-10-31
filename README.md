

#  Hybrid CNN-LSTM for Solar Irradiance Forecasting 

[![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)](https://www.python.org/)  [![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12-orange?logo=tensorflow)](https://www.tensorflow.org/)  [![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)  ![Last Update](https://img.shields.io/badge/Last%20Update-Aug%202025-brightgreen)

>  A deep learning-based approach for **short-term solar irradiance forecasting** using a **Hybrid CNN-LSTM** model with infrared image processing, inspired by cutting-edge research.

---

##  Project Overview

Accurate solar irradiance forecasting is critical for **renewable energy planning** and **grid stability** .  

This project implements a **Hybrid CNN-LSTM model** that:
- 🖼️ Uses **Convolutional Neural Networks (CNNs)** for spatial feature extraction  
- ⏳ Uses **Long Short-Term Memory (LSTM)** for temporal prediction  
- 🔬 Leverages **infrared satellite imagery** and **NASA POWER datasets**

---

##  Methodology

###  Steps Involved
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

##  Project Structure

```

Hybrid-CNN-LSTM-for-Solar-Irradiance-Forecasting/
│
# Hybrid CNN-LSTM for Solar Irradiance Forecasting

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)  
![Last Update](https://img.shields.io/badge/Last%20Update-Oct%202025-brightgreen)

Professional, concise overview of the repository: a PyTorch implementation of hybrid CNN-LSTM models for short-term solar irradiance forecasting. The project includes preprocessing utilities, training scripts, evaluation notebooks, and a FastAPI-based inference service.

---

## Key features

- Modular PyTorch implementations for CNN, LSTM, and hybrid architectures
- Robust checkpoint loading utilities to handle mismatched or weights-only checkpoints
- FastAPI inference service with templates and static assets for a simple UI
- Notebooks for training, evaluation and diagnostics

---

## Repository layout

```
./
├── APP/                        # legacy/alternate app folder
├── app.py / app1.py / app2.py  # older Flask app(s) and experimental runners
├── Fast_api_app.py             # Single-file FastAPI runner (for local testing)
├── app_fastapi/                # Modular FastAPI package (main, routes, utils, models, schemas)
├── scripts/                    # Model definitions, dataset helpers, preprocessing
├── notebooks/                  # Jupyter notebooks (training, evaluation, utilities)
├── models/                     # Saved PyTorch checkpoints (.pth)
├── models1/                    # alternate saved models / experiment checkpoints
├── logs/                       # Training history and evaluation logs (json)
├── data/                       # Processed inputs used by notebooks and training
├── GIRASOL_DATASET/            # Raw dataset folders used for reproducibility
├── templates/                  # Jinja2 templates used by the web UI
├── static/                     # CSS/JS/asset files for the UI
├── uploads/                    # temporary uploaded files used by the app
├── Samples/                    # example inputs / sample images
├── docs/                       # documentation and notes
├── Training/                   # training experiments or saved training scripts
├── requirements.txt            # Python dependencies (generated from venv)
├── README.md                   # this file
├── LICENSE
└── METHOD.md                   # project methodology notes
```

---

## Tech stack

- Python 3.9+ (venv)
- PyTorch (modeling, checkpoints)
- FastAPI + Uvicorn (inference API)
- Jupyter / notebooks
- OpenCV, NumPy, pandas for preprocessing and data handling

---

## Installation (Windows / PowerShell)

1. Create and activate a venv (PowerShell):

```powershell
python -m venv .\venv
.\venv\Scripts\Activate.ps1   # or use Activate.bat for cmd.exe
python -m pip install --upgrade pip
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

Note: `requirements.txt` was generated from the development venv and may include packages used during experimentation. Consider creating a trimmed `requirements-runtime.txt` for production.

---

## Quick start — run the API (development)

Run the single-file FastAPI runner (quick local test):

```powershell
python .\Fast_api_app.py
```

Recommended — run the modular app with uvicorn (hot reload):

```powershell
python -m uvicorn "app_fastapi.main:app" --reload --host 127.0.0.1 --port 8000
```

Then open http://127.0.0.1:8000/ in your browser.

---

## Typical workflows

- Preprocess data:
   - `python scripts/preprocess.py`
- Train models:
   - `python scripts/train_cnn.py`
   - `python scripts/train_lstm.py`
- Evaluate models / run inference:
   - use `notebooks/` for experiments or `scripts/evaluate_models.py` for scripted runs

---

## Contributing

Contributions are welcome. Suggested steps for contributors:

1. Fork the repository and create a feature branch
2. Run tests and notebooks in a matching venv/PyTorch environment
3. Open a pull request with a clear description and relevant change examples

For larger changes (API, deployment), please open an issue first to discuss design.

---

## License

This project is licensed under the MIT License — see `LICENSE` for details.

---

## Contact

Anand Bhimagouda Patil — ap6272440@gmail.com

- FastAPI + Uvicorn (inference API)
- Jupyter / notebooks
- OpenCV, NumPy, pandas for preprocessing and data handling

---

## Installation (Windows / PowerShell)

1. Create and activate a venv (PowerShell):

```powershell
python -m venv .\venv
.\venv\Scripts\Activate.ps1   # or use Activate.bat for cmd.exe
python -m pip install --upgrade pip
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

Note: `requirements.txt` in this repository was generated from the development venv and may include packages used during experimentation. Consider trimming it for production deployments.

---

## Running the FastAPI server

Run the single-file runner (quick local test):

```powershell
python .\Fast_api_app.py
```

Or run the modular app with uvicorn (recommended during development):

```powershell
python -m uvicorn "app_fastapi.main:app" --reload --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000/ in your browser.

---

## Notebooks and evaluation

- Use the notebooks in `notebooks/` to reproduce training and evaluation experiments. Several notebooks include robust model-loading helpers that attempt to recover mismatched checkpoints.
- If you run notebooks on a different machine, ensure the PyTorch/CUDA versions match the checkpoint expectations.

---

## Common workflows

- Preprocess data:
  - python scripts/preprocess.py
- Train (examples):
  - python scripts/train_cnn.py
  - python scripts/train_lstm.py
- Evaluate / run inference (examples in `scripts/evaluate_models.py` and notebooks)

---

## Contributing & notes

- If you prepare a production deployment (Docker, systemd, cloud), create a minimal runtime `requirements.txt` or a container image with only necessary dependencies.
- Run tests and notebooks in a matching Python & PyTorch environment when possible.

---

## License

This project is licensed under the MIT License — see `LICENSE`.

---

## Author

**Anand Bhimagouda Patil** — ap6272440@gmail.com


