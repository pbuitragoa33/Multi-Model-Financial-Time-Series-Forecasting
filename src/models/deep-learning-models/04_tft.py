# Temporal Fusion Transformer (TFT) - Simplified Implementation
# Architecture: LSTM Encoder-Decoder + Multi-Head Attention + Gating
# Features: Handles Past Observed Inputs & Known Future Inputs (Time Embeddings)


# Necessary imports


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset



# Path configurations

load_dotenv()

prepared_data_path = os.getenv('PREPARED_DATA_PATH')      
eval_pred_tft_path = os.getenv('OUT_EVAL_PRED_20_TFT')  
objects_path = os.getenv('OUT_OBJECTS_PATH')
dl_models_path = os.getenv('OUT_MODEL_PATH_DL')    

prepared_data_path = Path(prepared_data_path)
eval_pred_tft_path = Path(eval_pred_tft_path)
objects_path = Path(objects_path)
dl_models_path = Path(dl_models_path)


# ----------------------
TARGET = "Close_SPY"
# ----------------------

# ---------------------------------------------------------------------
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# ----------------------------------------------------------------------



# -------------------------------------- Early Stopping Class --------------------------------------

# Early Stopping to prevent overfitting (The class build was suggested by Copilot)

class EarlyStopping:
    
    def __init__(self, patience = 700, min_delta = 0):

        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):

        if self.best_loss is None:

            self.best_loss = val_loss

        elif val_loss > self.best_loss - self.min_delta:

            self.counter += 1

            if self.counter >= self.patience:

                self.early_stop = True

        else:
            self.best_loss = val_loss
            self.counter = 0



# -------------------------------------- Functions --------------------------------------


# 1. Add Time Features --> Generates known future calendat variables

def add_time_features(df):

    df = df.copy()

    if not isinstance(df.index, pd.DatetimeIndex):

        df.index = pd.to_datetime(df.index)
        
    # Cyclical variables and normalized ordinals

    df['day_of_week'] = df.index.dayofweek / 6.0       
    df['day_of_month'] = df.index.day / 31.0          
    df['month'] = df.index.month / 12.0                
    
    # List of time feature columns

    time_cols = ['day_of_week', 'day_of_month', 'month']

    return df, time_cols


# 2. Load data

def load_data():

    print("Loading and Augmenting Data with Time Features...")
    
    train_df = pd.read_csv(prepared_data_path / 'train_dataset.csv', index_col = 0, parse_dates = True)
    val_df = pd.read_csv(prepared_data_path / 'validation_dataset.csv',  index_col = 0, parse_dates = True)
    test_df = pd.read_csv(prepared_data_path / 'test_dataset.csv',  index_col = 0, parse_dates = True)

    scaler_target = joblib.load(objects_path / 'target_robust_scaler.joblib')
    
    # Add Time Features (Future known inputs)

    train_df, time_cols = add_time_features(train_df)
    val_df, ux= add_time_features(val_df)
    test_df, ux = add_time_features(test_df)
    
    # Separate targets

    train_y = train_df[TARGET].values
    val_y = val_df[TARGET].values
    test_y = test_df[TARGET].values

    # Separate Features
    # PAST_X: All (Economic Features + Time Features + Target Lagged if required)
    # FUTURE_X: Just Time Features 
    
    # Drop target from features

    feature_cols = [c for c in train_df.columns if c != TARGET]
    
    train_past_X = train_df[feature_cols].values
    val_past_X = val_df[feature_cols].values
    test_past_X = test_df[feature_cols].values
    
    # For the future known inputs, only time features

    train_future_X = train_df[time_cols].values
    val_future_X = val_df[time_cols].values
    test_future_X = test_df[time_cols].values
    
    # Dates

    dates = {'train': train_df.index, 'val': val_df.index, 'test': test_df.index}

    return (train_y, val_y, test_y, 
            train_past_X, val_past_X, test_past_X, 
            train_future_X, val_future_X, test_future_X, 
            scaler_target, dates)


# 3. Sequence Creation for TFT

def create_tft_sequences(past_X, future_X, y, seq_length, horizon):

    enc_inputs = []
    dec_inputs = []
    targets = []
    
    for i in range(len(past_X) - seq_length - horizon + 1):

        # 1. Past Observed Data

        enc_seq = past_X[i : i + seq_length]
        
        # 2. Future Known Data (Time features only)
        # Note: The decoder needs to know the future inputs for the horizon period

        dec_seq = future_X[i + seq_length : i + seq_length + horizon]
        
        # 3. Target

        y_seq = y[i + seq_length : i + seq_length + horizon]
        
        enc_inputs.append(enc_seq)
        dec_inputs.append(dec_seq)
        targets.append(y_seq)
        
    return (torch.tensor(np.array(enc_inputs), dtype = torch.float32), 
            torch.tensor(np.array(dec_inputs), dtype = torch.float32), 
            torch.tensor(np.array(targets), dtype = torch.float32))


# 4. DataLoaders Creation for TFT

def create_dataloaders(data_tuple, seq_len, horizon, batch_size):

    (tr_y, val_y, te_y, tr_px, val_px, te_px, tr_fx, val_fx, te_fx, _, _) = data_tuple
    
    # Train, Validation and Test (Arrays -> TensorDataset -> DataLoader)

    tr_enc, tr_dec, tr_tgt = create_tft_sequences(tr_px, tr_fx, tr_y, seq_len, horizon)
    val_enc, val_dec, val_tgt = create_tft_sequences(val_px, val_fx, val_y, seq_len, horizon)
    te_enc, te_dec, te_tgt = create_tft_sequences(te_px, te_fx, te_y, seq_len, horizon)

    train_loader = DataLoader(TensorDataset(tr_enc, tr_dec, tr_tgt), batch_size = batch_size, shuffle = False)
    val_loader = DataLoader(TensorDataset(val_enc, val_dec, val_tgt), batch_size = batch_size, shuffle = False)
    test_loader = DataLoader(TensorDataset(te_enc, te_dec, te_tgt), batch_size = batch_size, shuffle = False)
    
    return train_loader, val_loader, test_loader


# 5. TFT Model Definition (Simplified)

class GatedLinearUnit(nn.Module):

    def __init__(self, input_size, hidden_size):

        super(GatedLinearUnit, self).__init__()
        self.fc = nn.Linear(input_size, hidden_size * 2)
        
    def forward(self, x):

        val = self.fc(x)
        val, gate = val.chunk(2, dim = -1)

        return val * torch.sigmoid(gate)


class TFTModel(nn.Module):

    def __init__(self, past_input_dim, future_input_dim, hidden_size, output_horizon, dropout = 0.2):

        super(TFTModel, self).__init__()
        
        # 1. Processing Variables (Embeddings/Projections)

        self.past_embedding = nn.Linear(past_input_dim, hidden_size)
        self.future_embedding = nn.Linear(future_input_dim, hidden_size)
        
        # 2. LSTM Encoder (Past Processing)

        self.encoder_lstm = nn.LSTM(hidden_size, hidden_size, batch_first = True, num_layers = 1)
        
        # 3. LSTM Decoder (Future Processing with Context)

        self.decoder_lstm = nn.LSTM(hidden_size, hidden_size, batch_first = True, num_layers = 1)
        
        # 4. Multi-Head Attention (Connecting Past to Future) --> Capture long-term dependencies

        self.attention = nn.MultiheadAttention(embed_dim = hidden_size, num_heads = 8, batch_first = True, dropout = dropout)
        
        # 5. Post-Attention Gating

        self.post_att_gate = GatedLinearUnit(hidden_size, hidden_size)
        self.residual_norm = nn.LayerNorm(hidden_size)
        
        # 6. Final Output --> Prediction layer (single value per time step)

        self.fc_out = nn.Linear(hidden_size, 1) 

    def forward(self, x_past, x_future):

        # x_past: (batch, seq_len, past_features)
        # x_future: (batch, horizon, future_features)
        
        # A. Embeddings

        past_emb = F.elu(self.past_embedding(x_past))      # (batch, seq_len, hidden)
        future_emb = F.elu(self.future_embedding(x_future)) # (batch, horizon, hidden)
        
        # B. Encoder LSTM

        enc_out, (hn, cn) = self.encoder_lstm(past_emb)
        
        # C. Decoder LSTM ---> Uses final state of encoder as initial state

        dec_out, _ = self.decoder_lstm(future_emb, (hn, cn))
        
        # D. Attention Mechanism
        # Query: Decoder (Future), Key/Value: Encoder (Past)

        attn_out, attn_weights = self.attention(query = dec_out, key = enc_out, value = enc_out)
        
        # E. Residual & Gating (Skip connection)

        x = self.residual_norm(dec_out + self.post_att_gate(attn_out))
        
        # F. Final Prediction  -> Linear Layer (batch, horizon, 1)

        prediction = self.fc_out(x) 
        
        return prediction.squeeze(-1), attn_weights


# 6. Train the TFT model

def train_tft_model(model, train_loader, val_loader, epochs, lr, patience):

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    early_stopper = EarlyStopping(patience = patience)
    
    train_losses = [] 
    val_losses = []
    
    print(f"Training TFT for {epochs} epochs...")
    
    for epoch in range(epochs):

        # Training

        model.train()
        batch_losses = []
        
        for enc, dec, tgt in train_loader:

            enc, dec, tgt = enc.to(DEVICE), dec.to(DEVICE), tgt.to(DEVICE)
            
            optimizer.zero_grad()
            preds, _ = model(enc, dec)
            loss = criterion(preds, tgt)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())
            
        t_loss = np.mean(batch_losses)
        train_losses.append(t_loss)
        
        # Validation

        model.eval()

        val_batch_losses = []

        with torch.no_grad():

            for enc, dec, tgt in val_loader:

                enc, dec, tgt = enc.to(DEVICE), dec.to(DEVICE), tgt.to(DEVICE)
                preds, _ = model(enc, dec)
                loss = criterion(preds, tgt)
                val_batch_losses.append(loss.item())
        
        v_loss = np.mean(val_batch_losses)
        val_losses.append(v_loss)
        
        if epoch % 50 == 0:

            print(f"Epoch {epoch}/{epochs} | Train: {t_loss:.5f} | Val: {v_loss:.5f}")
            
        early_stopper(v_loss)

        if early_stopper.early_stop:

            print("Early Stopping triggered")
            print(f"Early Stopping at epoch {epoch}")

            break
            
    torch.save(model.state_dict(), dl_models_path / '20_tft_model.pth')

    return train_losses, val_losses


# 7. Evaluate the LSTM model and graphics

def get_predictions_and_attention(model, loader, scaler):

    model.eval()

    preds_list, true_list, attn_list = [], [], []
    
    with torch.no_grad():

        for enc, dec, tgt in loader:

            enc, dec = enc.to(DEVICE), dec.to(DEVICE)
            out, attn = model(enc, dec)
            
            preds_list.append(out.cpu().numpy())
            true_list.append(tgt.numpy())
            attn_list.append(attn.cpu().numpy()) # (batch, horizon, seq_len) bear in mind
            
    preds = np.concatenate(preds_list)
    trues = np.concatenate(true_list)
    attns = np.concatenate(attn_list)
    
    # Denormalize predictions and true values

    N, H = preds.shape

    preds_inv = scaler.inverse_transform(preds.reshape(-1, 1)).reshape(N, H)
    trues_inv = scaler.inverse_transform(trues.reshape(-1, 1)).reshape(N, H)
    
    return preds_inv, trues_inv, attns


# 8. Plot Attention Heatmap

def plot_attention_heatmap(attns, save_path):

    # Average attention across all samples
    # attns shape: (N_samples, Horizon, Seq_Length)
    # What past days matter most

    avg_attn = np.mean(attns, axis = 0) 
    
    plt.figure(figsize = (28, 24))

    sns.heatmap(avg_attn, cmap = 'plasma', cbar_kws = {'label': 'Attention Weight'})
    plt.title('TFT Attention Map: Past Sequence vs Future Horizon')
    plt.xlabel('Past Sequence Steps (0 = Oldest, 60 = Recent)')
    plt.ylabel('Future Horizon Steps (0 = Tomorrow)')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# 9. Generate Plots for TFT

def generate_plots_tft(tr_loss, val_loss, y_true, y_pred, future_pred, dates_test, dates_future, attns):
    
    # 1. Loss

    plt.figure(figsize = (28, 24))

    plt.plot(tr_loss, label = 'Train', color = 'royalblue', linewidth = 2)
    plt.plot(val_loss, label = 'Val',  color = 'orangered', linewidth = 2)
    plt.title('TFT Training Convergence', fontsize = 28)
    plt.savefig(eval_pred_tft_path / 'tft_loss.png')
    plt.close()
    
    # 2. Scatter Plot

    plt.figure(figsize = (28, 24))

    plt.scatter(y_true.flatten(), y_pred.flatten(), alpha = 0.3, color = 'darkslategray', s = 2)

    min_v, max_v = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())

    plt.plot([min_v, max_v], [min_v, max_v], 'r--')
    plt.title('Accuracy Scatter', fontsize = 28)
    plt.savefig(eval_pred_tft_path / 'tft_scatter.png')
    plt.close()
    
    # 3. Time Series Clean (t+1 forecast)

    valid_dates = dates_test[-len(y_true):]

    plt.figure(figsize = (28, 24))

    plt.plot(valid_dates, y_true[:, 0], label = 'Real Market Price',  color = 'black', linewidth = 2)
    plt.plot(valid_dates, y_pred[:, 0], label = 'TFT Forecast (t + 1)', color = 'springgreen', linewidth = 2, linestyle = '--')
    plt.title('TFT One-Step Ahead Forecast (Test Set)', fontsize = 28)
    plt.ylabel('SPY Price USD', fontsize = 24)
    plt.legend(fontsize = 18)
    plt.grid(alpha = 0.3)
    plt.savefig(eval_pred_tft_path / 'tft_timeseries.png')
    plt.close()
    
    # 4. Future Forecast with History Context (90 days history + 30 days future)

    hist_len = 90
    hist_y = y_true[-hist_len:, 0]
    hist_dates = np.arange(hist_len)
    fut_dates_idx = np.arange(hist_len, hist_len + len(future_pred))
    
    plt.figure(figsize = (28, 24))

    plt.plot(hist_dates, hist_y, label = 'Recent History (90d)', color = 'black', linewidth = 2)
    plt.plot(fut_dates_idx, future_pred, label = 'TFT Future Projection (30d)', color = 'red', marker = 'o', markersize = 4)

    plt.title('TFT Future Market Projection (30 days)', fontsize = 28)
    plt.legend(fontsize = 18)
    plt.grid(alpha = 0.3)
    plt.savefig(eval_pred_tft_path / 'tft_future_forecast.png')
    plt.close()
    
    # 5. Attention Heatmap 

    plot_attention_heatmap(attns, eval_pred_tft_path / 'tft_attention_map.png')




# ------------------------------------------------------------------------------------------------------------


# Main


if __name__ == '__main__':

    print("TFT Pipeline Running...")
    
    # Configurations

    # seq_length: Number of past timesteps to use as input featuress
    # horizon: Number of future timesteps to predict
    # batch_size: Number of samples per gradient update
    # epochs: Number of training epochs or cycles
    # patience: Early Stopping limit for validation loss improvement

    seq_length = 60
    horizon = 30
    batch_size = 32
    epochs = 2000
    lr = 1e-6
    patience = 1600
    hidden_size = 64
    

    # 1. Load Data

    data_tuple = load_data()
    
    (tr_y, _, te_y, tr_px, _, te_px, tr_fx, _, te_fx, scaler, dates) = data_tuple
    
    # 2. DataLoaders

    train_dl, val_dl, test_dl = create_dataloaders(data_tuple, seq_length, horizon, batch_size)
    
    # 3. Init Model
    # input_dim past = features size (economics)
    # input_dim future = time features size (3: day_week, day_month, month)

    past_dim = tr_px.shape[1]
    future_dim = tr_fx.shape[1]
    
    model = TFTModel(past_dim, future_dim, hidden_size, horizon).to(DEVICE)
    
    # 4. Train

    t_loss, v_loss = train_tft_model(model, train_dl, val_dl, epochs, lr, patience)
    
    # 5. Eval

    model.load_state_dict(torch.load(dl_models_path / '20_tft_model.pth'))

    pred_test, true_test, attns = get_predictions_and_attention(model, test_dl, scaler)
    
    rmse = np.sqrt(mean_squared_error(true_test, pred_test))
    mae = mean_absolute_error(true_test, pred_test)
    r2 = r2_score(true_test, pred_test)

    print(f"TFT RMSE: ${rmse:.4f}")
    print(f"TFT MAE: ${mae:.4f}")
    print(f"TFT R2 Score: {r2:.4f}")
    
    # 6. Future Forecast

    last_past_seq = torch.tensor(te_px[-seq_length:], dtype = torch.float32).unsqueeze(0).to(DEVICE)
    
    last_date = dates['test'][-1]
    future_dates = pd.bdate_range(start = last_date + pd.Timedelta(days = 1), periods = horizon)
    
    future_df_temp = pd.DataFrame(index = future_dates, data = {'dummy': range(horizon)})
    future_df_feat, _ = add_time_features(future_df_temp)
    future_time_feats = future_df_feat[['day_of_week', 'day_of_month', 'month']].values
    
    last_future_seq = torch.tensor(future_time_feats, dtype = torch.float32).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():

        future_scaled, _ = model(last_past_seq, last_future_seq)
        
    future_real = scaler.inverse_transform(future_scaled.cpu().numpy().reshape(-1, 1)).flatten()
    
    # Save CSV

    pd.DataFrame({'Date': future_dates, 'Predicted_TFT': future_real}).to_csv(eval_pred_tft_path / 'tft_future.csv')
    
    # 7. Plots

    print("Generating TFT Visualizations...")

    generate_plots_tft(t_loss, v_loss, true_test, pred_test, future_real, dates['test'], dates['test'], attns)
    
    print("TFT Pipeline Finished.")