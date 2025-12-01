import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import pickle
from liquidnet.main import LiquidNet # Make sure to install this

# --- 1. Dataset Configuration ---
DATASET_CONFIG = {
    'FD001': {
        'model_path': 'best_tuned_model_FD001.pth',
        'scaler_path': 'scaler_fd001.pkl',
        'static_cols': ['op_setting_3', 'sensor_1', 'sensor_10', 'sensor_18', 'sensor_19'],
        'num_features': 19, # 24 features - 5 static
        'window_size': 40,
        'cnn': 48,
        'lnn': 150,
        'dense': 72
    },
    'FD002': {
        'model_path': 'best_tuned_model_FD002.pth',
        'scaler_path': 'scaler_fd002.pkl',
        'static_cols': [],
        'num_features': 24,
        'window_size': 40,
        'cnn': 48,
        'lnn': 150,
        'dense': 72
    },
    'FD003': {
        'model_path': 'best_tuned_model_FD003.pth',
        'scaler_path': 'scaler_fd003.pkl',
        'static_cols': ['op_setting_3', 'sensor_1', 'sensor_18', 'sensor_19'],
        'num_features': 20,
        'window_size': 40,
        'cnn': 48,
        'lnn': 150,
        'dense': 72
    },
    'FD004': {
        'model_path': 'best_tuned_model_FD004.pth',
        'scaler_path': 'scaler_fd004.pkl',
        'static_cols': [],
        'num_features': 24,
        'window_size': 40,
        'cnn': 48,
        'lnn': 150,
        'dense': 72
    }
}


# --- 2. Model & Preprocessing Definitions ---

class HybridCnnLnn(nn.Module):
    def __init__(self, num_features, cnn_filters, lnn_units, dense_units):
        super(HybridCnnLnn, self).__init__()
        self.lnn_units = lnn_units
        self.cnn_feature_extractor = nn.Sequential(
            nn.Conv1d(in_channels=num_features, out_channels=cnn_filters, kernel_size=5, padding='same'),
            nn.ReLU(), nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=cnn_filters, out_channels=cnn_filters * 2, kernel_size=3, padding='same'),
            nn.ReLU(), nn.MaxPool1d(kernel_size=2), nn.Dropout(0.5))
        self.lnn_core = LiquidNet(num_units=lnn_units)
        self.output_head = nn.Sequential(
            nn.Linear(in_features=lnn_units, out_features=dense_units),
            nn.ReLU(), nn.Linear(in_features=dense_units, out_features=1))

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.cnn_feature_extractor(x)
        x = x.permute(0, 2, 1)
        batch_size, seq_len = x.size(0), x.size(1)
        hidden_state = torch.zeros(batch_size, self.lnn_units).to(x.device)
        for t in range(seq_len):
            output, hidden_state = self.lnn_core(x[:, t, :], hidden_state)
        final_prediction = self.output_head(hidden_state)
        return final_prediction

@st.cache_resource
def load_model_and_scaler(dataset_name):
    config = DATASET_CONFIG[dataset_name]
    model = HybridCnnLnn(
        num_features=config['num_features'],
        cnn_filters=config['cnn'],
        lnn_units=config['lnn'],
        dense_units=config['dense']
    )
    dummy_input = torch.zeros(1, config['window_size'], config['num_features'])
    model(dummy_input)
    model.load_state_dict(torch.load(config['model_path'], map_location=torch.device('cpu')))
    model.eval()
    
    with open(config['scaler_path'], 'rb') as f:
        scaler = pickle.load(f)
        
    st.success(f"Loaded model and scaler for {dataset_name}.")
    return model, scaler

# --- NEW: Two Preprocessing Functions ---

def preprocess_for_all_engines(df_raw, scaler, config):
    WINDOW_SIZE = config['window_size']
    static_cols = config['static_cols']
    all_feature_cols = [col for col in df_raw.columns if col not in ['engine_id', 'cycle']]
    
    df_processed = df_raw.drop(columns=static_cols)
    feature_cols_to_scale = [col for col in all_feature_cols if col not in static_cols]
    num_features = len(feature_cols_to_scale)
    
    df_processed[feature_cols_to_scale] = scaler.transform(df_processed[feature_cols_to_scale])
    
    sequences = []
    for engine_id in df_processed['engine_id'].unique():
        engine_df = df_processed[df_processed['engine_id'] == engine_id]
        feature_data = engine_df[feature_cols_to_scale].tail(WINDOW_SIZE).values
        
        if feature_data.shape[0] < WINDOW_SIZE:
            padding = np.zeros((WINDOW_SIZE - feature_data.shape[0], num_features))
            feature_data = np.concatenate((padding, feature_data), axis=0)
            
        sequences.append(feature_data)
        
    return torch.from_numpy(np.array(sequences)).float()

def preprocess_for_single_engine(df_raw, engine_id, scaler, config):
    WINDOW_SIZE = config['window_size']
    static_cols = config['static_cols']
    
    engine_df = df_raw[df_raw['engine_id'] == engine_id]
    if engine_df.empty:
        return None

    all_feature_cols = [col for col in df_raw.columns if col not in ['engine_id', 'cycle']]
    df_processed = engine_df.drop(columns=static_cols)
    feature_cols_to_scale = [col for col in all_feature_cols if col not in static_cols]
    num_features = len(feature_cols_to_scale)

    df_processed[feature_cols_to_scale] = scaler.transform(df_processed[feature_cols_to_scale])
    
    feature_data = df_processed[feature_cols_to_scale].tail(WINDOW_SIZE).values
    
    if feature_data.shape[0] < WINDOW_SIZE:
        padding = np.zeros((WINDOW_SIZE - feature_data.shape[0], num_features))
        feature_data = np.concatenate((padding, feature_data), axis=0)
        
    X_single = torch.from_numpy(np.array([feature_data])).float()
    return X_single

# --- 3. Build the Streamlit Web App ---
st.set_page_config(layout="wide")
st.title('Aircraft Engine Predictive Maintenance')
st.write("Using a Hybrid CNN-LNN Model to Predict Remaining Useful Life (RUL)")

# Dataset Selector
dataset_name = st.selectbox(
    "1. Select the dataset model:",
    ('FD001', 'FD002', 'FD003', 'FD004')
)

config = DATASET_CONFIG[dataset_name]
model, scaler = load_model_and_scaler(dataset_name)

# File Uploader
uploaded_file = st.file_uploader(f"2. Upload the corresponding test file (e.g., test_{dataset_name}.txt)")

if uploaded_file is not None:
    column_names = ['engine_id', 'cycle'] + [f'op_setting_{i}' for i in range(1, 4)] + [f'sensor_{i}' for i in range(1, 22)]
    df_test_raw = pd.read_csv(uploaded_file, sep='\\s+', header=None, names=column_names)
    
    st.write(f"File loaded successfully. Processing all {len(df_test_raw['engine_id'].unique())} engines...")

    # --- Full Report (Runs Automatically) ---
    X_test_all = preprocess_for_all_engines(df_test_raw, scaler, config)
    
    with torch.no_grad():
        predictions_all = model(X_test_all)
        
    results_df = pd.DataFrame({
        'Engine ID': df_test_raw['engine_id'].unique(),
        'Predicted RUL (Cycles)': predictions_all.numpy().flatten()
    })
    
    st.subheader("Inspection Report (All Engines)")
    INSPECTION_THRESHOLD = 30
    low_rul_engines = results_df[results_df['Predicted RUL (Cycles)'] < INSPECTION_THRESHOLD]
    
    if not low_rul_engines.empty:
        st.error(f"**Action Required!** {len(low_rul_engines)} engines require immediate inspection (RUL < {INSPECTION_THRESHOLD}).")
        st.dataframe(low_rul_engines.sort_values(by='Predicted RUL (Cycles)').style.format({'Predicted RUL (Cycles)': '{:.2f}'}))
    else:
        st.success(f"**All Clear!** No engines are predicted to fail within the next {INSPECTION_THRESHOLD} cycles.")

    st.subheader("Full Prediction Results (All Engines)")
    st.dataframe(results_df.style.format({'Predicted RUL (Cycles)': '{:.2f}'}))
    
    st.divider()

    # --- Single Engine Query ---
    st.subheader("Query a Single Engine")
    engine_id_to_check = st.number_input("Enter Engine ID:", min_value=1, step=1, value=1)
    
    if st.button(f"Predict RUL for Engine {engine_id_to_check}"):
        
        X_single = preprocess_for_single_engine(df_test_raw, engine_id_to_check, scaler, config)
        
        if X_single is None:
            st.error(f"Engine ID {engine_id_to_check} not found in the uploaded file.")
        else:
            with torch.no_grad():
                prediction = model(X_single)
            
            rul_value = prediction.numpy().flatten()[0]
            
            st.metric(label=f"Engine {engine_id_to_check} Predicted RUL", value=f"{rul_value:.2f} cycles")
            
            if rul_value < INSPECTION_THRESHOLD:
                st.error(f"**Status:** Engine {engine_id_to_check} requires immediate inspection.")
            else:
                st.success(f"**Status:** Engine {engine_id_to_check} is operating normally.")