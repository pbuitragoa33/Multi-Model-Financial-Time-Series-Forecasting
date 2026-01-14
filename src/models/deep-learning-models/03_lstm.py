# LSTM (Long Short-Term Memory) 


# Data preparation --> data load and data loader building
# Model building --> model training
# Model evaluating --> model testing (RMSE, MAE, R2 as metrics)
# Model saving in .pth
# Model predicting --> future forecasting


# Necessary imports

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from dotenv import load_dotenv
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# Path configurations

load_dotenv()

prepared_data_path = os.getenv('PREPARED_DATA_PATH')      
eval_pred_lstm_path = os.getenv('OUT_EVAL_PRED_19_LSTM')  
objects_path = os.getenv('OUT_OBJECTS_PATH')
dl_models_path = os.getenv('OUT_MODEL_PATH_DL')    

prepared_data_path = Path(prepared_data_path)
eval_pred_lstm_path = Path(eval_pred_lstm_path)
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
    
    def __init__(self, patience = 350, min_delta = 0):

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


# 1. Load data

def load_data():

    print("Loading subsets (train, validation, test) and target scaler")
    
    train_df = pd.read_csv(prepared_data_path / 'train_dataset.csv', index_col = 0, parse_dates = True)
    val_df = pd.read_csv(prepared_data_path / 'validation_dataset.csv',  index_col = 0, parse_dates = True)
    test_df = pd.read_csv(prepared_data_path / 'test_dataset.csv',  index_col = 0, parse_dates = True)

    scaler_target = joblib.load(objects_path / 'target_robust_scaler.joblib')

    
    # Target series (Close_SPY scaled)

    train_y = train_df[TARGET].values
    val_y   = val_df[TARGET].values
    test_y  = test_df[TARGET].values

    # Features variables (all other columns)

    train_X = train_df.drop(columns = [TARGET]).values
    val_X   = val_df.drop(columns = [TARGET]).values
    test_X  = test_df.drop(columns = [TARGET]).values

    # Dates for plotting later

    dates = {
        'train': train_df.index,
        'val': val_df.index,
        'test': test_df.index
    }

    return train_y, val_y, test_y, train_X, val_X, test_X, scaler_target, dates


# 2. Create multistep sequences for LSTM input

def create_sequences(X, y, seq_length, horizon):

    xs = []
    ys = []

    for i in range(len(X) - seq_length - horizon + 1):

        x_seq = X[i: i + seq_length]
        y_seq = y[i + seq_length: i + seq_length + horizon]

        xs.append(x_seq)
        ys.append(y_seq)

    # Return tensors

    return (
        torch.tensor(np.array(xs), dtype = torch.float32),
        torch.tensor(np.array(ys), dtype = torch.float32)
        )
    

# 3. Create DataLoaders

def create_dataloaders(train_X, train_y, val_X, val_y, test_X, test_y, seq_length, horizon, batch_size):

    # Invoke create_sequences function

    X_train_seq, y_train_seq = create_sequences(train_X, train_y, seq_length, horizon)
    X_val_seq, y_val_seq = create_sequences(val_X, val_y, seq_length, horizon)
    X_test_seq, y_test_seq = create_sequences(test_X, test_y, seq_length, horizon)

    # Crrate TensorDatasets

    train_dataset = TensorDataset(X_train_seq, y_train_seq)
    val_dataset = TensorDataset(X_val_seq, y_val_seq)
    test_dataset = TensorDataset(X_test_seq, y_test_seq)

    # Create DataLoaders

    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = False)
    val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle = False)
    test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False)

    return train_loader, val_loader, test_loader


# 4. Define the LSTM model

class LSTMModel(torch.nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout = 0.2):

        super(LSTMModel, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first = True, dropout = dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):

        out, (h_n, c_n) = self.lstm(x)  # (batch_size, seq_length, hidden_size) 
        last_time_step = out[:, -1, :]   # Take the last timestep output
        prediction = self.fc(last_time_step)   # (batch_size, output_size = horizon)

        return prediction


# 5. Train the LSTM model

def train_lstm_model(model, train_loader, val_loader, num_epochs, learning_rate, weight_decay, patience = 350):

    # Define loss function and optimizer

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, weight_decay = weight_decay)
    early_stopper = EarlyStopping(patience = patience)

    train_losses = []
    val_losses = []
    
    print(f"Training LSTM for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):

        # Training

        model.train()

        batch_losses = []

        for X_b, y_b in train_loader:

            X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)

            optimizer.zero_grad()
            y_pred = model(X_b)
            loss = criterion(y_pred, y_b)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())
        
        train_loss = np.mean(batch_losses)
        train_losses.append(train_loss)
        
        # Validation

        model.eval()

        val_batch_losses = []

        with torch.no_grad():

            for X_v, y_v in val_loader:

                X_v, y_v = X_v.to(DEVICE), y_v.to(DEVICE)

                pred_v = model(X_v)
                loss_v = criterion(pred_v, y_v)
                val_batch_losses.append(loss_v.item())
        
        val_loss = np.mean(val_batch_losses)
        val_losses.append(val_loss)
        
        if epoch % 50 == 0:

            print(f"   Epoch {epoch}/{num_epochs} | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f}")
            
        # Check Early Stopping

        early_stopper(val_loss)

        if early_stopper.early_stop:

            print("Early Stopping Activated")
            print(f"Early Stopping triggered at epoch {epoch}")

            break
            
    # Save the trained model

    torch.save(model.state_dict(), dl_models_path / '19_lstm_model.pth')

    return train_losses, val_losses

 
# 6. Evaluate the LSTM model and graphics

# Get predictions and obtain the re-scaled values
# full_preds: Complete (N, Horizon) matrix for metrics
# step1_preds: Only the first horizon step predicted (N, 1) for cleaner plots

def get_predictions(model, loader, scaler):

    model.eval()

    preds_list = []
    true_list = []
    
    with torch.no_grad():

        for X_b, y_b in loader:

            X_b = X_b.to(DEVICE)
            out = model(X_b).cpu().numpy()
            preds_list.append(out)
            true_list.append(y_b.numpy())
            
    preds = np.concatenate(preds_list) # (N, Horizon)
    trues = np.concatenate(true_list)  # (N, Horizon)
    
    # De-scale predictions and true values
    # The scaler expects (N, 1). Since we have (N, Horizon), we flatten, de-scale, and reshape
    
    N, H = preds.shape

    preds_inv = scaler.inverse_transform(preds.reshape(-1, 1)).reshape(N, H)
    trues_inv = scaler.inverse_transform(trues.reshape(-1, 1)).reshape(N, H)
    
    return preds_inv, trues_inv


# 7. Generate evaluation plots

def generate_plots(train_losses, val_losses, y_true, y_pred, train_true, train_pred, future_pred, dates_test, dates_future):
    
    # Plot 1: Loss Curve

    plt.figure(figsize = (28, 24))

    plt.plot(train_losses, label = 'LSTM Train Loss', color = 'royalblue', linewidth = 2)
    plt.plot(val_losses, label = 'LSTM Validation Loss', color = 'orangered', linewidth = 2)
    plt.title('LSTM Model: Loss over Epochs', fontsize = 28)
    plt.xlabel('Epoch', fontsize = 24)
    plt.ylabel('MSE Loss', fontsize = 24)
    plt.legend(fontsize = 18)
    plt.grid(True, alpha = 0.3)
    plt.savefig(eval_pred_lstm_path / 'lstm_loss_curve.png')
    plt.close()
    
    # Plot 2: Accuracy Scatter Plot

    plt.figure(figsize = (28, 24))

    plt.scatter(y_true.flatten(), y_pred.flatten(), alpha = 0.3, color = 'darkslategray', s = 2)
    
    # Line of perfect prediction

    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())

    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth = 3, label = 'Perfect Prediction')
    
    plt.title('Accuracy: Predicted vs Real Values (All Horizons)', fontsize = 28)
    plt.xlabel('Real Price USD', fontsize = 24)
    plt.ylabel('Predicted Price USD', fontsize = 24)
    plt.legend(fontsize = 18)
    plt.grid(True, alpha = 0.3)
    plt.savefig(eval_pred_lstm_path / 'lstm_scatter.png')
    plt.close()
    
    # Plot 3: Test Set Time Series 
    # Just one-step ahead for clarity 
    # Date adjustment to match y_true length

    valid_dates = dates_test[-len(y_true):]
    
    plt.figure(figsize = (28, 24))

    plt.plot(valid_dates, y_true[:, 0], label = 'Real Market Price', color = 'black', linewidth = 2)
    plt.plot(valid_dates, y_pred[:, 0], label = 'LSTM Forecast (t + 1)', color = 'springgreen', linewidth = 2, linestyle = '--')
    plt.title('LSTM Test Set Performance: One-Step Ahead Forecast', fontsize = 28)
    plt.ylabel('SPY Price USD', fontsize = 24)
    plt.legend(fontsize = 18)
    plt.grid(True, alpha = 0.3)
    plt.xticks(rotation = 45)
    plt.tight_layout()
    plt.savefig(eval_pred_lstm_path / 'lstm_test_series.png')
    plt.close()
    
    # Plot 4: Training Fit - Model Learning

    plt.figure(figsize = (28, 24))

    plt.plot(train_true[:, 0], label = 'Real Train Data', color = 'black', linewidth = 2)
    plt.plot(train_pred[:, 0], label = 'Model Fit', color = 'deepskyblue', alpha = 0.85, linewidth = 2)
    plt.title('Training Set Fit (LSTM Model Learning Capability)', fontsize = 28)
    plt.legend(fontsize = 18)
    plt.grid(True, alpha = 0.3)
    plt.savefig(eval_pred_lstm_path / 'lstm_train_fit.png')
    plt.close()
    
    # Plot 5: Future Projection

    plt.figure(figsize = (28, 24))
    
    # Last 60 days from test subset
    
    last_60_days = y_true[-60:, 0]
    hist_dates = np.arange(len(last_60_days))

    # Future prediction (30 days)

    future_x = np.arange(len(last_60_days), len(last_60_days) + len(future_pred))
    
    plt.plot(hist_dates, last_60_days, label = 'Recent History (60d)', color = 'black', linewidth = 2)
    plt.plot(future_x, future_pred, label = 'LSTM Future Projection (30d)', color = 'red', marker = 'o', markersize = 4)
    
    # Connect the last historical point to the first future prediction

    plt.plot([hist_dates[-1], future_x[0]], [last_60_days[-1], future_pred[0]], 'r--')
    
    plt.title('Future Market Projection', fontsize = 28)
    plt.xlabel('Trading Days (Relative)', fontsize = 24)
    plt.ylabel('SPY Price USD', fontsize = 24)
    plt.legend(fontsize = 18)
    plt.grid(True, alpha = 0.3)
    plt.savefig(eval_pred_lstm_path / 'lstm_future_forecast.png')
    plt.close()




# ------------------------------------------------------------------------------------------------------------


# Main

if __name__ == '__main__':

    print("LSTM Pipeline...")

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
    learning_rate = 1e-6  
    weight_decay = 1e-3   
    patience = 1500


    # 1. Load data

    y_train, y_val, y_test, X_train, X_val, X_test, scaler_target, dates = load_data()

    # 2. Create DataLoaders

    train_dl, val_dl, test_dl = create_dataloaders(X_train, y_train, X_val, y_val, X_test, y_test, seq_length, horizon, batch_size) 

    # 3. LSTM Model instantiation

    input_size = X_train.shape[1]

    model = LSTMModel(
        input_size = input_size,
        hidden_size = 128,
        num_layers = 2,
        output_size = horizon,
        dropout = 0.2,
    ).to(DEVICE)

    # 4. Model training

    train_losses, val_losses = train_lstm_model(model, train_dl, val_dl, epochs, learning_rate, weight_decay, patience)

    # 5. Model evaluation 

    model.load_state_dict(torch.load(dl_models_path / '19_lstm_model.pth'))
    print("Evaluating LSTM model on test set and generating plots...")

    pred_test, true_test = get_predictions(model, test_dl, scaler_target)
    pred_train, true_train = get_predictions(model, train_dl, scaler_target)

    # Metrics

    mse = mean_squared_error(true_test, pred_test)

    rmse = np.sqrt(mse)
    mae = mean_absolute_error(true_test, pred_test)
    r2 = r2_score(true_test, pred_test)

    print("--" * 25)
    print("LSTM Model Metrics on Test Set:")
    print("LSTM RMSE: ", round(rmse, 3))
    print("LSTM MAE: ", round(mae, 3))
    print("LSTM R2: ", round(r2, 3))

    print("--" * 25)


    # 6. Future forecasting (30 days ahead) with last sequence from test set

    last_sequence_tensor = torch.tensor(X_test[-seq_length:], dtype = torch.float32).unsqueeze(0).to(DEVICE)

    # Note: X_test[-seq_length:] is incorrect if X_test is not yet sequenced. 
    # Correction: We need the last seq_length raw data points from X_test

    last_raw_seq = X_test[-seq_length:] # (60, features)
    last_seq_tensor = torch.tensor(last_raw_seq, dtype = torch.float32).unsqueeze(0).to(DEVICE)

    with torch.no_grad():

        future_scaled = model(last_seq_tensor).cpu().numpy().flatten()    

    future_real = scaler_target.inverse_transform(future_scaled.reshape(-1, 1)).flatten()

    # Generate future dates

    last_date = dates['test'][-1]
    future_dates = pd.bdate_range(start = last_date + pd.Timedelta(days = 1), periods = horizon)

    # Save as CSV file

    pd.DataFrame({'Date': future_dates, 'Predicted': future_real}).to_csv(eval_pred_lstm_path / 'lstm_future_predictions.csv')


    # 7. Generates all plots

    print("Generating plots...")

    generate_plots(train_losses, val_losses, true_test, pred_test, true_train, pred_train, future_real, dates['test'], future_dates)

    print("Pipeline Finished Successfully")