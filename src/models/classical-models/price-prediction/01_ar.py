# AR Model

# Load the data and the scaler.
# Find the best lag p using train + validation.
# Train the final AR(p) model using only train and save it to joblib.
# Evaluate the test, calculate RMSE, and generate a graph.
# Predict the next 30 days after the test, save a CSV file, and generate a projection graph.


# Necessary libraries

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import joblib
from pathlib import Path
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error


# Paths configuration

load_dotenv()

prepared_data_path = os.getenv('PREPARED_DATA_PATH')  # Load the data
target_scaler_path = os.getenv('OUT_OBJECTS_PATH')    # Load the target scaler

out_diverse_path = os.getenv('OUT_DIVERSE_PATH')      # Diverse savable items
out_model_path = os.getenv('OUT_MODEL_PATH_CLASSIC')  # Save the model as joblib
eval_pred_ar_path = os.getenv('OUT_EVAL_PRED_01_AR')  # Save the prediction (csv and image)

prepared_data_path = Path(prepared_data_path)
target_scaler_path = Path(target_scaler_path)
out_diverse_path = Path(out_diverse_path)
out_model_path = Path(out_model_path)
eval_pred_ar_path = Path(eval_pred_ar_path)


# ----------------------
TARGET = "Close_SPY"
# ----------------------



# -------------------------------------- Functions --------------------------------------


# 1. Load data

def load_data():

    print("Loading subsets (train, validation, test) and target scaler")
    
    train_df = pd.read_csv(prepared_data_path / 'train_dataset.csv', index_col = 0, parse_dates = True)
    val_df = pd.read_csv(prepared_data_path / 'validation_dataset.csv',  index_col = 0, parse_dates = True)
    test_df = pd.read_csv(prepared_data_path / 'test_dataset.csv',  index_col = 0, parse_dates = True)

    scaler_target = joblib.load(target_scaler_path / 'target_robust_scaler.joblib')

    
    # Manage business days ('B'). If there are NaN ffill() and later bfill()

    train = train_df[TARGET].asfreq("B").ffill()
    val   = val_df[TARGET].asfreq("B").ffill()
    test  = test_df[TARGET].asfreq("B").ffill()

    train = train.bfill()
    val = val.ffill()
    test = test.bfill()

    return train, val, test, scaler_target


# 2. Optimize p value (lag) using validation set

def search_best_p(train, val, p_min = 1, p_max = 30):

    mse_results = {}

    for p in range(p_min, p_max + 1):

        model = AutoReg(endog = train, lags = p, old_names = False).fit()

        preds_val = model.predict(start = val.index[0], end = val.index[-1])

        mse = mean_squared_error(y_true = val, y_pred = preds_val)
        mse_results[p] = mse

    best_p = min(mse_results, key = mse_results.get) 

    # Image p vs MSE

    plt.figure(figsize = (18, 12))

    plt.plot(list(mse_results.keys()), list(mse_results.values()), marker = "o")
    plt.title("MSE vs p (lags) — Validation Set")
    plt.xlabel("p (lag)")
    plt.ylabel("MSE")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_diverse_path/"01_ar_p_vs_mse.png")

    print(f"Save image")

    return best_p


# 3. Train final model using train set

def train_model(train, best_p):

    model = AutoReg(endog = train, lags = best_p, old_names = False).fit()

    joblib.dump(model, out_model_path / '01_ar_model.joblib')

    return model


# 4. Evaluate test: Predict under test set, calculate RMSE y generates an image

def evaluate_test(model, test, scaler_target):

    preds = model.predict(start = test.index[0], end = test.index[-1])

    preds_eval = scaler_target.inverse_transform(preds.values.reshape(-1, 1)).flatten()
    real_eval = scaler_target.inverse_transform(test.values.reshape(-1, 1)).flatten()

    rmse = np.sqrt(mean_squared_error(y_true = real_eval, y_pred = preds_eval))

    print("RMSE Best AR Model: ", round(rmse, 2))

    # Image Prediction vs Real values

    plt.figure(figsize = (18, 12))

    plt.plot(test.index, real_eval, label = 'Real (Test)', color = 'black', alpha = 0.7)
    plt.plot(test.index, preds_eval, label = f'Prediction AR({model.model._lags})', color = 'red')
    plt.title(f'Prediction vs Real — AR({model.model._lags})')
    plt.xlabel("Date")
    plt.ylabel("SPY Close Price")
    plt.grid(True, alpha = 0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(eval_pred_ar_path / "AR_test_prediction.png")


# 5. Future prediction with full datasets

def future_prediction(model, train, val, test, scaler_target, horizon = 30):

    # Concatenate the subsets
    
    full_history = pd.concat([train, val, test])

    # Recursive prediction

    last_index = full_history.index[-1]
    
    preds_future = []

    history_values = full_history.values.tolist()

    n_lags = len(model.model._lags)

    for i in range(horizon):

        input_array = history_values[-n_lags:]
        yhat = np.dot(model.params[1:], np.array(input_array[::-1])) + model.params[0]
        preds_future.append(yhat)
        history_values.append(yhat)

    # De-scale

    preds_future_val = scaler_target.inverse_transform(np.array(preds_future).reshape(-1, 1)).flatten()

    # Future dates (Business Days)

    last_date = full_history.index[-1]
    future_dates = pd.bdate_range(start = last_date + pd.Timedelta(days = 1), periods = horizon)

    # Save CSV

    future_df = pd.DataFrame({'Date': future_dates, 'Predicted_Close': preds_future_val})
    future_df.set_index('Date', inplace = True)

    future_df.to_csv(eval_pred_ar_path / 'future_30d_prediction.csv')

    print("Future prediction saved")

    # Plot the last 90 days of data + future prediction

    plt.figure(figsize = (22, 20))

    recent_history = full_history.tail(90).values.reshape(-1, 1)
    recent_history_spy = scaler_target.inverse_transform(recent_history).flatten()
    recent_dates = full_history.tail(90).index

    plt.plot(recent_dates, recent_history_spy, label = 'Recent History', color = 'black')
    plt.plot(future_dates, preds_future_val, label = f'Prediction AR({model.model._lags}) +{horizon}d', color = 'red', marker = 'o')
    plt.title(f'Prediction SPY Close Price')
    plt.xlabel('Date')
    plt.ylabel('SPY Close Price')
    plt.grid(True, alpha = 0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(eval_pred_ar_path / 'future_30d_forecast_chart.png')


# ------------------------------------------------------------------------------------------------------------

# Main 


if __name__ == "__main__":

    print("AR Model Pipeline...")

    # 1. Load data
    
    train, val, test, scaler_target = load_data()

    # 2. Optimal p-value (lag value)

    best_p = search_best_p(train, val)

    # 3. Model training

    model = train_model(train, best_p)

    # 4. Test evaluation

    evaluate_test(model, test, scaler_target)

    # 5. Future Prediction

    future_prediction(model, train, val, test, scaler_target, horizon = 30)

    print("Pipeline done")