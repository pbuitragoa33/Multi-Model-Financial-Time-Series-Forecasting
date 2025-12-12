# ARIMA Model

# Load the data and the scaler.
# Find the best combination (p, d, q) using train + validation.****
# Train the final ARIMA(p, d, q) model using only train and save it to joblib.****
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

prepared_data_path = os.getenv('PREPARED_DATA_PATH')  # Load the data
target_scaler_path = os.getenv('OUT_OBJECTS_PATH')    # Load the target scaler

out_diverse_path = os.getenv('OUT_DIVERSE_PATH')      # Diverse savable items
out_model_path = os.getenv('OUT_MODEL_PATH_CLASSIC')  # Save the model as joblib
eval_pred_arima_path = os.getenv('OUT_EVAL_PRED_04_ARIMA')  # Save the prediction (csv and image)

prepared_data_path = Path(prepared_data_path)
target_scaler_path = Path(target_scaler_path)
out_diverse_path = Path(out_diverse_path)
out_model_path = Path(out_model_path)
eval_pred_arima_path = Path(eval_pred_arima_path)


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
    val = val.bfill()
    test = test.bfill()

    return train, val, test, scaler_target


# 2. Search best ARIMA(p, d, q) using validation 

def search_best_pdq(train, val, p_max = 5, d_max = 2, q_max = 5):

    mse_results = {}

    history_train_val = pd.concat([train, val])

    for p in range(0, p_max + 1):

        for d in range(0, d_max + 1):

            for q in range(0, q_max + 1):

                if p == 0 and d == 0 and q == 0:

                    continue

                try: 

                    model = ARIMA(train, order = (p, d, q)).fit()

                    model_val = model.apply(history_train_val)
                    preds_val = model.forecast(steps = len(val))

                    mse = mean_squared_error(val, preds_val)
                    mse_results[(p, d, q)] = mse

                except Exception:

                    continue

    best_pdq = min(mse_results, key = mse_results.get)
    best_p, best_d, best_q = best_pdq

    print(f"Best ARIMA Order: (p = {best_p}, d = {best_d}, q = {best_q})")

    # Scatter plot (p, q) colored by MSE (d component are ignored for simplicity)
    
    plt.figure(figsize = (18, 12))

    x_vals = [k[0] for k in mse_results.keys()]
    y_vals = [k[2] for k in mse_results.keys()]
    z_vals = [v for v in mse_results.values()]

    plt.scatter(x_vals, y_vals, c = z_vals, cmap = "inferno")
    plt.colorbar(label = "MSE")
    plt.title("MSE for ARIMA(p, d, q) — Validation Set")
    plt.xlabel("p")
    plt.ylabel("q")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_diverse_path / "04_arima_pdq_vs_mse.png")

    return best_p, best_d, best_q


# 3. Train final ARIMA model

def train_model(train, best_p, best_d, best_q):

    model = ARIMA(train, order = (best_p, best_d, best_q)).fit()

    joblib.dump(model, out_model_path / '04_arima_model.joblib')

    return model


# 4. Evaluate test set

def evaluate_test(model, train, val, test, scaler_target):

    full_history = pd.concat([train, val, test])

    model_test = model.apply(full_history)

    preds_scaled = model_test.fittedvalues[test.index[0]:]

    preds_eval = scaler_target.inverse_transform(preds_scaled.values.reshape(-1, 1)).flatten()
    real_eval = scaler_target.inverse_transform(test.values.reshape(-1, 1)).flatten()

    rmse = np.sqrt(mean_squared_error(real_eval, preds_eval))

    print("RMSE Best ARMA Model: ", round(rmse, 2))

    # Image prediction vs Real values

    plt.figure(figsize = (18, 12))

    plt.plot(test.index, real_eval, label = "Real (Test)", color = "black", alpha = 0.7)
    plt.plot(test.index, preds_eval, label = f"Prediction ARIMA{model.model.order}", color = "orchid")
    plt.title(f"Prediction vs Real — ARIMA{model.model.order}")
    plt.xlabel("Date")
    plt.ylabel("SPY Close Price")
    plt.grid(True, alpha = 0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(eval_pred_arima_path / "ARIMA_test_prediction.png")


# 5. Future prediction with dataset concatened

def future_prediction(model, train, val, test, scaler_target, horizon = 30):

    full_data = pd.concat([train, val, test])

    final_model_fit = ARIMA(full_data, order = model.model.order).fit()

    preds_future_scaled = final_model_fit.forecast(steps = horizon)

    preds_future_val = scaler_target.inverse_transform(preds_future_scaled.values.reshape(-1, 1)).flatten()

    last_date = full_data.index[-1]
    future_dates = pd.bdate_range(start = last_date + pd.Timedelta(days = 1), periods = horizon)

    future_df = pd.DataFrame({"Date": future_dates, "Predicted_Close": preds_future_val})
    future_df.set_index("Date", inplace = True)

    future_df.to_csv(eval_pred_arima_path / "future_30d_prediction_ARIMA.csv")

    print("Future prediction saved")

    # Plot last 90 days + future projection

    plt.figure(figsize = (22, 20))

    recent_history = full_data.tail(90).values.reshape(-1, 1)
    recent_history_usd = scaler_target.inverse_transform(recent_history).flatten()
    recent_dates = full_data.tail(90).index

    plt.plot(recent_dates, recent_history_usd, label = "Recent History", color = "black")
    plt.plot(future_dates, preds_future_val,
             label = f"Prediction ARIMA{model.model.order} +{horizon}d",
             color = "orchid", marker = "o")

    plt.title("SPY Close Price — Future Projection")
    plt.xlabel("Date")
    plt.ylabel("SPY Close Price")
    plt.grid(True, alpha = 0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(eval_pred_arima_path / "future_30d_projection_ARIMA.png")



# ------------------------------------------------------------------------------------------------------------

# Main

if __name__ == '__main__':

    print("ARIMA Model Pipeline...")
    
    # 1. Load data

    train, val, test, scaler_target = load_data()

    # 2. Optimal p-value, d-value and q-value

    best_p, best_d, best_q = search_best_pdq(train, val)

    # 3. Model training

    model = train_model(train, best_p, best_d, best_q)

    # 4. Test evaluation

    evaluate_test(model, train, val, test, scaler_target)

    # 5. Future Prediction

    future_prediction(model, train, val, test, scaler_target, horizon = 30)

    print("Pipeline done")