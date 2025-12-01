# ✈️ Smart Wings for Safer Skies: Predictive Maintenance of Aircraft Engines

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B)
![License](https://img.shields.io/badge/License-MIT-green)

> **A Hybrid Deep Learning Approach using 1D-CNN and Liquid Neural Networks (LNN) for Remaining Useful Life (RUL) Prediction.**

---

##  Table of Contents
- [Project Overview](#-project-overview)
- [The Dataset](#-the-dataset)
- [Methodology & Architecture](#-methodology--architecture)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
  - [Training the Model](#1-training-the-model)
  - [Running the Web App](#2-running-the-web-app)
- [Results](#-results)
- [Contributors](#-contributors)

---

##  Project Overview
Predictive Maintenance (PdM) is critical in aviation to prevent catastrophic failures and reduce maintenance costs. This project implements a data-driven framework to predict the **Remaining Useful Life (RUL)** of turbofan jet engines.

Moving beyond traditional Recurrent Neural Networks (like LSTMs), this project proposes a **Hybrid Architecture**:
1.  **1D-CNN:** For automatic feature extraction from multivariate time-series sensor data.
2.  **Liquid Neural Network (LNN):** A novel, adaptive recurrent neural network modeled by differential equations, capable of handling non-stationary and noisy data streams more effectively than static RNNs.

The system is trained on the NASA C-MAPSS benchmark and deployed via a user-friendly **Streamlit Interface** for real-time engine health monitoring.

---

##  The Dataset
We utilized the **NASA Commercial Modular Aero-Propulsion System Simulation (C-MAPSS)** dataset. It consists of four sub-datasets with varying complexity:

| Dataset | Train Trajectories | Test Trajectories | Operational Conditions | Fault Modes | Complexity |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **FD001** | 100 | 100 | 1 | 1 | Lowest |
| **FD002** | 260 | 259 | 6 | 1 | High (Complex Ops) |
| **FD003** | 100 | 100 | 1 | 2 | Medium (Complex Faults) |
| **FD004** | 249 | 248 | 6 | 2 | Highest (Real-world sim) |

*Input Data:* 21 sensors (Temperature, Pressure, Fan Speeds) and 3 operational settings.

---

##  Methodology & Architecture

The pipeline consists of three main stages:

### 1. Data Preprocessing
* **RUL Calculation:** Derived target labels from run-to-failure cycles.
* **RUL Clipping:** Capped max RUL at **125 cycles** to focus learning on the critical degradation phase.
* **Feature Selection:** Removed static sensor columns with zero variance.
* **Normalization:** Applied `MinMaxScaler` to scale sensor readings to [0, 1].
* **Sliding Window:** Generated 3D sequences with a **Window Size of 40**.

### 2. The Hybrid Model
* **Input Layer:** Takes sequence shape `(Batch, 40, Features)`.
* **Feature Extractor (1D-CNN):** Two convolutional layers with ReLU activation and Max Pooling to identify local sensor patterns.
* **Temporal Core (LNN):** A `LiquidNet` layer (based on ODEs) processes the CNN output to capture long-term temporal dependencies and degradation trends.
* **Regressor (Output Head):** Fully connected layers map the LNN state to a single continuous RUL value.

### 3. Evaluation Metrics
* **RMSE:** Root Mean Squared Error (Precision).
* **Asymmetric C-MAPSS Score:** A domain-specific metric that penalizes late predictions (false negatives) more heavily than early ones.

---

##  Project Structure

```text
├── Capstone(LNN).ipynb      # Main Jupyter Notebook (Training, Tuning, Evaluation)
├── app.py                   # Streamlit Web Application for inference
├── requirements.txt         # List of dependencies
├── data/                    # Folder containing raw train/test text files
│   ├── train_FD001.txt
│   └── ...
├── models/                  # Saved PyTorch Models (.pth)
│   ├── best_tuned_model_FD001.pth
│   └── ...
├── scalers/                 # Saved Data Scalers (.pkl)
│   ├── scaler_fd001.pkl
│   └── ...
└── README.md                # Project Documentation

##  Installation

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/YourUsername/Aircraft-Predictive-Maintenance.git](https://github.com/YourUsername/Aircraft-Predictive-Maintenance.git)
    cd Aircraft-Predictive-Maintenance
    ```

2.  **Create a Virtual Environment (Optional but Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *Note: Ensure `liquidnet` is installed. If not available via pip, please refer to the installation cell in the notebook.*

---

##  Usage

### 1. Training the Model
To reproduce the training results or retrain on new data:
1.  Open `Capstone(LNN).ipynb` in Jupyter Lab, Google Colab, or VS Code.
2.  Ensure dataset files are in the correct path.
3.  Run the cells sequentially. The notebook handles data loading, preprocessing, training loop, and saving the best models/scalers.

### 2. Running the Web App
To launch the interactive dashboard:
```bash
python -m streamlit run app.py