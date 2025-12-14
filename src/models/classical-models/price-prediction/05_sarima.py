# SARIMA Model

# Load the data and the scaler.
# Find the best combination (p, d, q) × (P, D, Q, s) using train + validation.****
# Train the final SARIMA model using only train and save it to joblib.****
# Evaluate on test, calculate RMSE, and generate a graph.
# Predict the next 30 days after the test, save a CSV file, and generate a projection graph.


# Necessary libraries

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import joblib
from pathlib import Path
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error



# Paths configuration

load_dotenv()

prepared_data_path = os.getenv('PREPARED_DATA_PATH')            # Load the data
target_scaler_path = os.getenv('OUT_OBJECTS_PATH')              # Load the target scaler

out_diverse_path = os.getenv('OUT_DIVERSE_PATH')                # Diverse savable items
out_model_path = os.getenv('OUT_MODEL_PATH_CLASSIC')            # Save the model as joblib
eval_pred_sarima_path = os.getenv('OUT_EVAL_PRED_05_SARIMA')    # Save the prediction (csv and image)

prepared_data_path = Path(prepared_data_path)
target_scaler_path = Path(target_scaler_path)
out_diverse_path = Path(out_diverse_path)
out_model_path = Path(out_model_path)
eval_pred_sarima_path = Path(eval_pred_sarima_path)



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


# 2. Search best SARIMA(p, d, q) × (P, D, Q) with fixed seasonality (s = 5 --> business days - market open week)

def search_best_sarima_hyperparms(train, val, p_max = 3, d_max = 1, q_max = 3, 
                                  P_max = 2, D_max = 1, Q_max = 2, s = 5):
    
    print("SARIMA hyperparameters...")

    mse_results = {}

    history_train_val = pd.concat([train, val])

    for p in range(0, p_max + 1):

        for d in range(0, d_max + 1):

            for q in range(0, q_max + 1):

                for P in range(0, P_max + 1):

                    for D in range(0, D_max + 1):

                        for Q in range(0, Q_max + 1):

                            if (p, d, q) == (0, 0, 0) and (P, D, Q) == (0, 0, 0):

                                continue

                            try:

                                model = SARIMAX(train, 
                                                order = (p, d, q),
                                                seasonal_order = (P, D, Q, s),
                                                enforce_stationarity = False,
                                                enforce_invertibility = False).fit(disp = False)
                                

                                model_val = model.apply(history_train_val)
                                preds_val = model.forecast(steps = len(val))

                                mse = mean_squared_error(val, preds_val)
                                mse_results[(p, d, q, P, D, Q)] = mse

                            except Exception:

                                continue

    best_params = min(mse_results, key = mse_results.get)
    best_p, best_d, best_q, best_P, best_D, best_Q = best_params

    print(f"Best SARIMA order: (p = {best_p}, d = {best_d}, q = {best_q}) × (P = {best_P}, D = {best_D}, Q = {best_Q}, s = {s})")


    # Multi-Subplots: MSE vs Parameters (AI generated - Gemini 3 Pro suggestion)

    fig, axes = plt.subplots(3, 2, figsize = (28, 30))
    axes = axes.flatten()

    mse_vals = list(mse_results.values())
    params = list(mse_results.keys())

    # 1 — p vs P

    ax = axes[0]
    ax.scatter([k[0] for k in params], [k[3] for k in params], c = mse_vals, cmap = "viridis")
    ax.set_title("p vs P — MSE")
    ax.set_xlabel("p")
    ax.set_ylabel("P")
    ax.grid(True)

    # 2 — p vs q

    ax = axes[1]
    ax.scatter([k[0] for k in params], [k[2] for k in params], c = mse_vals, cmap = "viridis")
    ax.set_title("p vs q — MSE")
    ax.set_xlabel("p")
    ax.set_ylabel("q")
    ax.grid(True)

    # 3 — d vs D

    ax = axes[2]
    ax.scatter([k[1] for k in params], [k[4] for k in params], c = mse_vals, cmap = "viridis")
    ax.set_title("d vs D — MSE")
    ax.set_xlabel("d")
    ax.set_ylabel("D")
    ax.grid(True)

    # 4 — q vs Q

    ax = axes[3]
    ax.scatter([k[2] for k in params], [k[5] for k in params], c = mse_vals, cmap = "viridis")
    ax.set_title("q vs Q — MSE")
    ax.set_xlabel("q")
    ax.set_ylabel("Q")
    ax.grid(True)

    # 5 — (p,d,q) flattened vs MSE

    ax = axes[4]
    pdq_index = [i for i in range(len(params))]
    ax.scatter(pdq_index, mse_vals, c = mse_vals, cmap = "viridis")
    ax.set_title("(p,d,q) combinations — MSE")
    ax.set_xlabel("Combination Index")
    ax.set_ylabel("MSE")
    ax.grid(True)

    # 6 — (P,D,Q) flattened vs MSE

    ax = axes[5]
    PDQ_index = [i for i in range(len(params))]
    ax.scatter(PDQ_index, mse_vals, c = mse_vals, cmap = "viridis")
    ax.set_title("(P,D,Q) combinations — MSE")
    ax.set_xlabel("Combination Index")
    ax.set_ylabel("MSE")
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(out_diverse_path / "05_sarima_multiplot_mse.png")
    plt.close()


    return best_p, best_d, best_q, best_P, best_D, best_Q, s


# 3. Train final SARIMA model

def train_model(train, best_p, best_d, best_q, best_P, best_D, best_Q, s):

    model = SARIMAX(train, 
                    order = (best_p, best_d, best_q), 
                    seasonal_order = (best_P, best_D, best_Q, s),
                    enforce_stationarity = False,
                    enforce_invertibility = False).fit(disp = False)
    
    joblib.dump(model, out_model_path / '05_sarima_model.joblib')

    return model


# 4. Evaluate on test

def evaluate_test(model, train, val, test, scaler_target):

    full_history = pd.concat([train, val, test])

    model_test = model.apply(full_history)
    preds_scaled = model_test.fittedvalues[test.index[0]:]

    preds_eval = scaler_target.inverse_transform(preds_scaled.values.reshape(-1, 1)).flatten()
    real_eval  = scaler_target.inverse_transform(test.values.reshape(-1, 1)).flatten()

    rmse = np.sqrt(mean_squared_error(real_eval, preds_eval))
    
    print("RMSE Best SARIMA Model:", round(rmse, 2))

    # Image Real vs Predictions

    plt.figure(figsize = (18, 12))

    plt.plot(test.index, real_eval, label = "Real (Test)", color = "black", alpha = 0.7)
    plt.plot(test.index, preds_eval, label = f"Prediction SARIMA", color = "royalblue")

    plt.title("Prediction vs Real — SARIMA")
    plt.xlabel("Date")
    plt.ylabel("SPY Close Price")
    plt.grid(True, alpha = 0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(eval_pred_sarima_path / "SARIMA_test_prediction.png")


# 5. Future Prediction

def future_prediction(model, train, val, test, scaler_target, horizon = 30):

    full_data = pd.concat([train, val, test])

    final_model_fit = SARIMAX(
        full_data,
        order = model.model.order,
        seasonal_order = model.model.seasonal_order,
        enforce_stationarity = False,
        enforce_invertibility = False).fit(disp = False)

    preds_future_scaled = final_model_fit.forecast(steps = horizon)
    preds_future_val = scaler_target.inverse_transform(preds_future_scaled.values.reshape(-1, 1)).flatten()

    last_date = full_data.index[-1]
    future_dates = pd.bdate_range(start = last_date + pd.Timedelta(days = 1), periods = horizon)

    df_future = pd.DataFrame({"Date": future_dates, "Predicted_Close": preds_future_val})
    df_future.set_index("Date", inplace = True)
    df_future.to_csv(eval_pred_sarima_path / "future_30d_prediction_SARIMA.csv")

    print("Future prediction saved")

    # Plot last 90 days + future prediction

    plt.figure(figsize = (22, 20))

    recent_history = full_data.tail(90).values.reshape(-1, 1)
    recent_history_usd = scaler_target.inverse_transform(recent_history).flatten()
    recent_dates = full_data.tail(90).index

    plt.plot(recent_dates, recent_history_usd, label = "Recent History", color = "black")
    plt.plot(future_dates, preds_future_val,
             label = f"Prediction SARIMA +{horizon}d",
             color = "royalblue", marker = "o")

    plt.title("SPY Close Price — Future Projection")
    plt.xlabel("Date")
    plt.ylabel("SPY Close Price")
    plt.grid(True, alpha = 0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(eval_pred_sarima_path / "future_30d_projection_SARIMA.png")



# ------------------------------------------------------------------------------------------------------------

# Main

if __name__ == '__main__':

    print("SARIMA Model Pipeline...")
    
    # 1. Loasd data

    train, val, test, scaler_target = load_data()

    # 2. Optimal SARIMA hyperparameters

    best_p, best_d, best_q, best_P, best_D, best_Q, s = search_best_sarima_hyperparms(train, val)

    # 3. Train model

    model = train_model(train, best_p, best_d, best_q, 
                        best_P, best_D, best_Q, s)  
    
    # 4. Test evaluation

    evaluate_test(model, train, val, test, scaler_target)

    # 5. Future prediction

    future_prediction(model, train, val, test, scaler_target, horizon = 30)

    print("Pipeline dodne")