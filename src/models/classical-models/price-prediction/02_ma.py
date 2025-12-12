# MA Model (Moving Average)

# Load the data and the scaler.
# Find the best q using train + validation.
# Train the final MA(q) model using only train and save it to joblib.
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
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error


# Paths configuration

load_dotenv()

prepared_data_path = Path(os.getenv('PREPARED_DATA_PATH'))   # Load the data
target_scaler_path = Path(os.getenv('OUT_OBJECTS_PATH'))     # Load the target scaler

out_diverse_path = Path(os.getenv('OUT_DIVERSE_PATH'))       # Diverse savable items
out_model_path = Path(os.getenv('OUT_MODEL_PATH_CLASSIC'))   # Save the model as joblib
eval_pred_ma_path = Path(os.getenv('OUT_EVAL_PRED_02_MA'))   # Save the prediction (csv and image)


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

    # Manage business days ('B'). If there are NaN ffill() and later bfill()

    train = train_df[TARGET].asfreq("B").ffill()
    val   = val_df[TARGET].asfreq("B").ffill()
    test  = test_df[TARGET].asfreq("B").ffill()

    train = train.bfill()
    val = val.bfill()
    test = test.bfill()

    return train, val, test, scaler_target


# 2. Optimize q (MA order) using validation set

def search_best_q(train, val, q_min = 1, q_max = 30):

    mse_results = {}

    history_train_val = pd.concat([train, val])

    for q in range(q_min, q_max + 1):

        model = ARIMA(train, order = (0, 1, q)).fit()

        model_val = model.apply(history_train_val)
        preds_val = model.forecast(steps = len(val))

        mse = mean_squared_error(val, preds_val)
        mse_results[q] = mse

    best_q = min(mse_results, key = mse_results.get)

    # Image q vs MSE

    plt.figure(figsize = (18, 12))

    plt.plot(list(mse_results.keys()), list(mse_results.values()), marker = "o")
    plt.title("MSE vs q (MA order) — Validation Set")
    plt.xlabel("q (order)")
    plt.ylabel("MSE")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_diverse_path / "02_ma_q_vs_mse.png")

    print("Save image")

    return best_q


# 3. Train final model

def train_model(train, best_q):

    model = ARIMA(train, order = (0, 1, best_q)).fit()

    joblib.dump(model, out_model_path / '02_ma_model.joblib')

    return model


# 4. Evaluate test set

def evaluate_test(model, train, val, test, scaler_target):

    full_history = pd.concat([train, val, test])

    model_test = model.apply(full_history)

    preds_scaled = model_test.fittedvalues[test.index[0]:]

    preds_eval = scaler_target.inverse_transform(preds_scaled.values.reshape(-1, 1)).flatten()
    real_eval  = scaler_target.inverse_transform(test.values.reshape(-1, 1)).flatten()

    rmse = np.sqrt(mean_squared_error(real_eval, preds_eval))

    print("RMSE Best MA Model: ", round(rmse, 2))

    # Image Prediction vs Real values

    plt.figure(figsize = (18, 12))

    plt.plot(test.index, real_eval, label = 'Real (Test)', color = 'black', alpha = 0.7)
    plt.plot(test.index, preds_eval, label = f'Prediction MA({model.model.k_ma})', color = 'green')
    plt.title(f'Prediction vs Real — MA({model.model.k_ma})')
    plt.xlabel("Date")
    plt.ylabel("SPY Close Price")
    plt.grid(True, alpha = 0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(eval_pred_ma_path / "MA_test_prediction.png")



# 5. Future prediction (30 days)

def future_prediction(model, train, val, test, scaler_target, horizon = 30):

    full_data = pd.concat([train, val, test])

    final_model_fit = ARIMA(full_data, order = model.model.order).fit()

    preds_future_scaled = final_model_fit.forecast(steps = horizon)

    preds_future_val = scaler_target.inverse_transform(preds_future_scaled.values.reshape(-1, 1)).flatten()

    last_date = full_data.index[-1]
    future_dates = pd.bdate_range(start = last_date + pd.Timedelta(days = 1), periods = horizon)

    future_df = pd.DataFrame({'Date': future_dates, 'Predicted_Close': preds_future_val})
    future_df.set_index('Date', inplace = True)
    
    future_df.to_csv(eval_pred_ma_path / 'future_30d_prediction_MA.csv')

    print("Future prediction saved")

    # Plot last 90 days + future projection

    plt.figure(figsize = (22, 20))

    recent_history = full_data.tail(90).values.reshape(-1, 1)
    recent_history_usd = scaler_target.inverse_transform(recent_history).flatten()
    recent_dates = full_data.tail(90).index

    plt.plot(recent_dates, recent_history_usd, label = 'Recent History', color = 'black')
    plt.plot(future_dates, preds_future_val, label = f'Prediction MA({model.model.k_ma}) +{horizon}d', color = 'green', marker = 'o')
    plt.title('Prediction SPY Close Price')
    plt.xlabel('Date')
    plt.ylabel('SPY Close Price')
    plt.grid(True, alpha = 0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(eval_pred_ma_path / 'future_30d_forecast_chart_MA.png')



# ------------------------------------------------------------------------------------------------------------

# Main

if __name__ == "__main__":

    print("MA Model Pipeline...")

    # 1. Load data

    train, val, test, scaler_target = load_data()

    # 2. Optimal q-value (order value)

    best_q = search_best_q(train, val)

    # 3. Model training

    model = train_model(train, best_q)

    # 4. Test evaluation

    evaluate_test(model, train, val, test, scaler_target)

    # 5. Future Prediction

    future_prediction(model, train, val, test, scaler_target, horizon = 30)

    print("Pipeline done")