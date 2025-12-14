# VAR Model

# Load the data and the scaler.
# Find the optimal lag order using train + validation.****
# Train the final VAR model using train only and save it to joblib.****
# Evaluate on test, calculate RMSE for target, and generate a graph.
# Predict the next 30 days after the test, save a CSV file, and generate a projection graph.


# Necessary libraries


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import joblib
from pathlib import Path
from statsmodels.tsa.api import VAR
from sklearn.metrics import mean_squared_error


# Paths configuration

load_dotenv()

prepared_data_path = os.getenv('PREPARED_DATA_PATH')
target_scaler_path = os.getenv('OUT_OBJECTS_PATH')

out_diverse_path = os.getenv('OUT_DIVERSE_PATH')
out_model_path = os.getenv('OUT_MODEL_PATH_CLASSIC')
eval_pred_var_path = os.getenv('OUT_EVAL_PRED_07_VAR')  

prepared_data_path = Path(prepared_data_path)
target_scaler_path = Path(target_scaler_path)
out_diverse_path = Path(out_diverse_path)
out_model_path = Path(out_model_path)
eval_pred_var_path = Path(eval_pred_var_path)



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

    # Fill missing dates with business frequency

    train = train_df.asfreq("B").ffill().bfill()
    val   = val_df.asfreq("B").ffill().bfill()
    test  = test_df.asfreq("B").ffill().bfill()

    return train, val, test, scaler_target


# 2. Select optimal lag order using train + validation (AIC, BIC, HQIC, FPE)

def select_best_lag(train, val, maxlags = 80):

    print("Selecting optimal VAR lag order...")
    print(f"Max lags tested: {maxlags}\n")

    data = pd.concat([train, val])
    
    # Check if data is sufficient for maxlags

    if len(data) < maxlags + 10:

        maxlags = len(data) - 10

        print(f"Data too short. Adjusted maxlags to: {maxlags}")

    model = VAR(data)

    # 1. Automatic Selection (AIC, BIC, etc.)

    try:

        lag_order_results = model.select_order(maxlags = maxlags)

        print("Lag Order Selection Criteria:")
        print("--------------------------------")
        print(f"AIC : {lag_order_results.selected_orders['aic']}")
        print(f"BIC : {lag_order_results.selected_orders['bic']}")
        print(f"HQIC: {lag_order_results.selected_orders['hqic']}")
        print(f"FPE : {lag_order_results.selected_orders['fpe']}")
        print("--------------------------------\n")
        
        # Default to AIC if available

        best_lag = lag_order_results.selected_orders["aic"]

    except Exception as e:

        print(f"Automatic selection failed: {e}")

        best_lag = 1 

    # 2. Manual Loop for Plotting
    # Initialize as LISTS to prevent "unsized object" errors

    aic_vals_list = []
    bic_vals_list = []
    hqic_vals_list = []
    fpe_vals_list = []
    
    # We iterate precisely over the range

    lags_range = range(1, maxlags + 1)

    for p in lags_range:

        try:

            res = model.fit(p)
            aic_vals_list.append(res.aic)
            bic_vals_list.append(res.bic)
            hqic_vals_list.append(res.hqic)
            fpe_vals_list.append(res.fpe)

        except Exception:

            aic_vals_list.append(np.nan)
            bic_vals_list.append(np.nan)
            hqic_vals_list.append(np.nan)
            fpe_vals_list.append(np.nan)

    # Convert to arrays for plotting

    aic_arr  = np.array(aic_vals_list)
    bic_arr  = np.array(bic_vals_list)
    hqic_arr = np.array(hqic_vals_list)
    fpe_arr  = np.array(fpe_vals_list)
    
    # Define lags array matching the length of collected metrics

    lags_arr = np.array(list(lags_range))

    # Plots
    fig, axes = plt.subplots(2, 2, figsize = (18, 14))

    # AIC

    axes[0, 0].plot(lags_arr, aic_arr, marker = "o")
    axes[0, 0].set_title("AIC vs Lag")

    if 'lag_order_results' in locals():

        axes[0, 0].axvline(lag_order_results.selected_orders["aic"], color = "red", ls = "--")

    axes[0, 0].grid(True, alpha = 0.3)

    # BIC

    axes[0, 1].plot(lags_arr, bic_arr, marker = "o")
    axes[0, 1].set_title("BIC vs Lag")

    if 'lag_order_results' in locals():

        axes[0, 1].axvline(lag_order_results.selected_orders["bic"], color = "red", ls = "--")

    axes[0, 1].grid(True, alpha = 0.3)

    # HQIC

    axes[1, 0].plot(lags_arr, hqic_arr, marker = "o")
    axes[1, 0].set_title("HQIC vs Lag")

    if 'lag_order_results' in locals():

        axes[1, 0].axvline(lag_order_results.selected_orders["hqic"], color = "red", ls = "--")

    axes[1, 0].grid(True, alpha = 0.3)

    # FPE

    axes[1, 1].plot(lags_arr, fpe_arr, marker = "o")
    axes[1, 1].set_title("FPE vs Lag")

    if 'lag_order_results' in locals():

        axes[1, 1].axvline(lag_order_results.selected_orders["fpe"], color = "red", ls = "--")

    axes[1, 1].grid(True, alpha = 0.3)

    fig.suptitle("VAR Lag Order Selection Criteria", fontsize = 18)
    plt.tight_layout(rect = [0, 0.03, 1, 0.95])
    
    out_diverse_path.mkdir(parents = True, exist_ok = True)
    plt.savefig(out_diverse_path / "07_var_lag_selection_criteria.png")
    plt.close()

    print(f"Selected lag order (AIC): {best_lag}\n")
    
    return best_lag


# 3. Train final VAR model

def train_model(train, best_lag):

    model = VAR(train)
    model_fitted = model.fit(best_lag)

    joblib.dump(model_fitted, out_model_path / '07_var_model.joblib')

    return model_fitted


# 4. Evaluate test set

def evaluate_test(model_fitted, train, val, test, scaler_target):

    full_data = pd.concat([train, val, test])
    lag_order = model_fitted.k_ar

    preds = model_fitted.forecast(full_data.values[-lag_order:], steps = len(test))
    preds_df = pd.DataFrame(preds, index = test.index, columns = full_data.columns)

    preds_eval = scaler_target.inverse_transform(preds_df[TARGET].values.reshape(-1, 1)).flatten()
    real_eval  = scaler_target.inverse_transform(test[TARGET].values.reshape(-1, 1)).flatten()

    rmse = np.sqrt(mean_squared_error(real_eval, preds_eval))

    print("RMSE VAR Model (target):", round(rmse, 2))

    # Plot

    plt.figure(figsize = (18, 12))

    plt.plot(test.index, real_eval, label = "Real (Test)", color = "black", alpha = 0.7)
    plt.plot(test.index, preds_eval, label = "Prediction VAR", color = "aqua")
    plt.title("VAR Prediction vs Real — Target")
    plt.xlabel("Date")
    plt.ylabel("SPY Close Price")
    plt.grid(True, alpha = 0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(eval_pred_var_path / "VAR_test_prediction.png")
    plt.close()


# 5. Future prediction 

def future_prediction(model_fitted, train, val, test, scaler_target, horizon = 30):

    full_data = pd.concat([train, val, test])
    lag_order = model_fitted.k_ar

    last_values = full_data.values[-lag_order:]

    preds_future = model_fitted.forecast(last_values, steps = horizon)
    preds_future_df = pd.DataFrame(preds_future, columns = full_data.columns)

    preds_future_val = scaler_target.inverse_transform(preds_future_df[TARGET].values.reshape(-1,1)).flatten()

    last_date = full_data.index[-1]
    future_dates = pd.bdate_range(start = last_date + pd.Timedelta(days = 1), periods = horizon)

    df_future = pd.DataFrame({"Date":future_dates, "Predicted_Close": preds_future_val})
    df_future.set_index("Date", inplace = True)
    df_future.to_csv(eval_pred_var_path / "future_30d_prediction_VAR.csv")

    print("Future prediction saved")

    # Plot last 90 days + future

    plt.figure(figsize = (22, 20))

    recent_history = full_data[TARGET].tail(90).values.reshape(-1, 1)
    recent_history_usd = scaler_target.inverse_transform(recent_history).flatten()
    recent_dates = full_data[TARGET].tail(90).index

    plt.plot(recent_dates, recent_history_usd, label = "Recent History", color = "black")
    plt.plot(future_dates, preds_future_val, label = f"Prediction VAR +{horizon}d", color = "aqua", marker = "o")
    plt.title("SPY Close Price — Future Projection VAR")
    plt.xlabel("Date")
    plt.ylabel("SPY Close Price")
    plt.grid(True, alpha = 0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(eval_pred_var_path / "future_30d_projection_VAR.png")
    plt.close()



# ------------------------------------------------------------------------------------------------------------


# Main

if __name__ == '__main__':

    print("VAR Model Pipeline...")

    # 1. Load data

    train, val, test, scaler_target = load_data()

    # 2. Select best lag

    best_lag = select_best_lag(train, val)

    # 3. Train model

    model_fitted = train_model(train, best_lag)

    # 4. Test evaluation

    evaluate_test(model_fitted, train, val, test, scaler_target)

    # 5. Future prediction
    
    future_prediction(model_fitted, train, val, test, scaler_target, horizon = 30)

    print("Pipeline done")
