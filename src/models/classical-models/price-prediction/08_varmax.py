# VARMAX Model

# Load the data and the scaler.
# Find the optimal (p, q) order using AIC on Train + Validation.
# Train the final VARMAX model using train only and save it.
# Evaluate on test, calculate RMSE for target, and generate a graph.
# Predict the next 30 days after the test, save a CSV file, and generate a projection graph.


# Necessary libraries

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
import joblib
from pathlib import Path
from statsmodels.tsa.statespace.varmax import VARMAX
from sklearn.metrics import mean_squared_error


# --- Paths configuration ---

load_dotenv()

prepared_data_path = os.getenv('PREPARED_DATA_PATH')
target_scaler_path = os.getenv('OUT_OBJECTS_PATH')

out_diverse_path = os.getenv('OUT_DIVERSE_PATH')
out_model_path = os.getenv('OUT_MODEL_PATH_CLASSIC')
eval_pred_varmax_path = os.getenv('OUT_EVAL_PRED_08_VARMAX')  

prepared_data_path = Path(prepared_data_path)
target_scaler_path = Path(target_scaler_path)
out_diverse_path = Path(out_diverse_path)
out_model_path = Path(out_model_path)
eval_pred_varmax_path = Path(eval_pred_varmax_path)


# ----------------------
TARGET = "Close_SPY"
# ----------------------



# -------------------------------------- Functions --------------------------------------


# 1. Load data

def load_data():

    print("Loading subsets (train, validation, test) and target scaler")

    train_df = pd.read_csv(prepared_data_path / 'train_dataset.csv', index_col = 0, parse_dates = True)
    val_df   = pd.read_csv(prepared_data_path / 'validation_dataset.csv', index_col = 0, parse_dates = True)
    test_df  = pd.read_csv(prepared_data_path / 'test_dataset.csv', index_col = 0, parse_dates = True)

    scaler_target = joblib.load(target_scaler_path / 'target_robust_scaler.joblib')

    train = train_df.asfreq("B").ffill().bfill()
    val   = val_df.asfreq("B").ffill().bfill()
    test  = test_df.asfreq("B").ffill().bfill()

    return train, val, test, scaler_target


# 2. Select optimal (p, q) order using AIC (Grid Search)

# No hyperparameter tuning, it's very complex and long


# 3. Train final VARMAX model

def train_model(train, order: tuple):

    print(f"Training final VARMAX{order} on training set...")
    
    model = VARMAX(train, order = (1, 1), trend = 'c', enforce_stationarity = False, enforce_invertibility = False)
    model_fitted = model.fit(disp = False, method = 'nm', maxiter = 1)
    
    joblib.dump(model_fitted, out_model_path / '08_varmax_model.joblib')
    
    return model_fitted


# 4. Evaluate test set

def evaluate_test(model_fitted, test, scaler_target):

    start_index = len(model_fitted.fittedvalues)
    end_index = start_index + len(test) - 1
    
    pred_res = model_fitted.get_forecast(steps = len(test))
    preds_df = pred_res.predicted_mean
    
    preds_df.index = test.index
    
    preds_eval = scaler_target.inverse_transform(preds_df[TARGET].values.reshape(-1, 1)).flatten()
    real_eval  = scaler_target.inverse_transform(test[TARGET].values.reshape(-1, 1)).flatten()

    rmse = np.sqrt(mean_squared_error(real_eval, preds_eval))

    print("RMSE VARMAX Model (target):", round(rmse, 2))

    # Plot

    plt.figure(figsize = (18, 16))

    plt.plot(test.index, real_eval, label = "Real (Test)", color = "black", alpha = 0.7)
    plt.plot(test.index, preds_eval, label = "Prediction VARMAX", color = "purple", linestyle = "--")
    plt.title(f"VARMAX Prediction vs Real — Target (RMSE: {rmse:.2f})")
    plt.xlabel("Date")
    plt.ylabel("SPY Close Price")
    plt.grid(True, alpha = 0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(eval_pred_varmax_path / "VARMAX_test_prediction.png")
    plt.close()


# 5. Future prediction 

def future_prediction(model_fitted, test, scaler_target, horizon = 30):

    
    total_steps = len(test) + horizon
    pred_res = model_fitted.get_forecast(steps = total_steps)
    full_preds = pred_res.predicted_mean
    
    future_preds_df = full_preds.iloc[-horizon:].copy()
    
    last_date = test.index[-1]
    future_dates = pd.bdate_range(start = last_date + pd.Timedelta(days = 1), periods = horizon)
    future_preds_df.index = future_dates
    
    preds_future_val = scaler_target.inverse_transform(future_preds_df[TARGET].values.reshape(-1, 1)).flatten()
    
    df_future = pd.DataFrame({"Date": future_dates, "Predicted_Close": preds_future_val})
    df_future.set_index("Date", inplace = True)
    df_future.to_csv(eval_pred_varmax_path / "future_30d_prediction_VARMAX.csv")

    # Plot

    plt.figure(figsize = (22, 20))
    
    recent_history = test[TARGET].tail(90).values.reshape(-1, 1)
    recent_history_usd = scaler_target.inverse_transform(recent_history).flatten()
    recent_dates = test.tail(90).index

    plt.plot(recent_dates, recent_history_usd, label = "Recent History (Test)", color = "black")
    plt.plot(future_dates, preds_future_val, label = f"Prediction VARMAX +{horizon}d", color = "purple", marker = "o")
    plt.title("SPY Close Price — Future Projection VARMAX")
    plt.xlabel("Date")
    plt.ylabel("SPY Close Price")
    plt.grid(True, alpha = 0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(eval_pred_varmax_path / "future_30d_projection_VARMAX.png")
    plt.close()

# ------------------------------------------------------------------------------------------------------------

# Main


if __name__ == '__main__':

    print("VARMAX Model Pipeline...")

    # 1. Load data

    train, val, test, scaler_target = load_data()

    # 3. Train model

    model_fitted = train_model(train, order = (1, 1))

    # 4. Test evaluation

    evaluate_test(model_fitted, test, scaler_target)

    # 5. Future prediction

    future_prediction(model_fitted, test, scaler_target, horizon = 30)

    print("\nPipeline done.")