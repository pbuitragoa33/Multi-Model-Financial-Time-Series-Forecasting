# SARIMAX Model

# Load the data and the scaler.
# Find the best combination (p, d, q) × (P, D, Q, s) using train + validation.****
# Train the final SARIMAX model using only train and save it to joblib.****
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
eval_pred_sarimax_path = os.getenv('OUT_EVAL_PRED_06_SARIMAX')    # Save the prediction (csv and image)

prepared_data_path = Path(prepared_data_path)
target_scaler_path = Path(target_scaler_path)
out_diverse_path = Path(out_diverse_path)
out_model_path = Path(out_model_path)
eval_pred_sarimax_path = Path(eval_pred_sarimax_path)



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

    
    # Target series

    train_y = train_df[TARGET].asfreq("B").ffill().bfill()
    val_y   = val_df[TARGET].asfreq("B").ffill().bfill()
    test_y  = test_df[TARGET].asfreq("B").ffill().bfill()

    # Exogenous variables (all other columns)

    train_X = train_df.drop(columns = [TARGET]).asfreq("B").ffill().bfill()
    val_X   = val_df.drop(columns = [TARGET]).asfreq("B").ffill().bfill()
    test_X  = test_df.drop(columns = [TARGET]).asfreq("B").ffill().bfill()

    return train_y, val_y, test_y, train_X, val_X, test_X, scaler_target


# 2. Search best SARIMA(p, d, q) × (P, D, Q) with fixed seasonality (s = 5 --> business days - market open week)

def search_best_sarimax_hyperparms(train_y, val_y, train_X, val_X, 
                                   p_max = 1, d_max = 1, q_max = 1,
                                   P_max = 1, D_max = 1, Q_max = 1, s = 5):
    
    print("SARIMAX hyperparameters...")

    mse_results = {}

    history_y = pd.concat([train_y, val_y])
    history_x = pd.concat([train_X, val_X])

    for p in range(0, p_max + 1):

        for d in range(0, d_max + 1):

            for q in range(0, q_max + 1):

                for P in range(0, P_max + 1):

                    for D in range(0, D_max + 1):

                        for Q in range(0, Q_max + 1):

                            if (p, d, q) == (0, 0, 0) and (P, D, Q) == (0, 0, 0):

                                continue

                            if d == 1 and D == 1:

                                continue

                            try:

                                model = SARIMAX(endog = train_y,
                                                exog = train_X,
                                                order = (p, d, q),
                                                seasonal_order = (P, D, Q, s),
                                                enforce_stationarity = False,
                                                enforce_invertibility = False).fit(disp = False)
                                
                                preds_val = model.forecast(steps = len(val_y), exog = val_X)

                                mse = mean_squared_error(val_y, preds_val)
                                mse_results[(p, d, q, P, D, Q)] = mse

                            except Exception:

                                continue

    best_params = min(mse_results, key = mse_results.get)
    best_p, best_d, best_q, best_P, best_D, best_Q = best_params

    print(f"Best SARIMAX order: (p = {best_p}, d = {best_d}, q = {best_q}) × (P = {best_P}, D = {best_D}, Q = {best_Q}, s = {s})")

    # Multi-Subplots: MSE vs Parameters (AI generated - Gemini 3 Pro suggestion)

    fig, axes = plt.subplots(3, 2, figsize = (26, 30))
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
    ax.scatter(list(range(len(params))), mse_vals, c = mse_vals, cmap = "viridis")
    ax.set_title("(p,d,q) combinations — MSE")
    ax.set_xlabel("Combination Index")
    ax.set_ylabel("MSE")
    ax.grid(True)

    # 6 — (P,D,Q) flattened vs MSE

    ax = axes[5]
    ax.scatter(list(range(len(params))), mse_vals, c = mse_vals, cmap = "viridis")
    ax.set_title("(P,D,Q) combinations — MSE")
    ax.set_xlabel("Combination Index")
    ax.set_ylabel("MSE")
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(out_diverse_path / "06_sarimax_multiplot_mse.png")
    plt.close()

    return best_p, best_d, best_q, best_P, best_D, best_Q, s


# 3. Train final SARIMAX model

def train_model(train_y, train_X, p, d, q, P, D, Q, s):

    model = SARIMAX(endog = train_y,
                    exog = train_X,
                    order = (p, d, q),
                    seasonal_order = (P, D, Q, s),
                    enforce_stationarity = False,
                    enforce_invertibility = False).fit(disp = False)

    joblib.dump(model,out_model_path / '06_sarimax_model.joblib')

    return model


# 4. Evaluate on test

def evaluate_test(model, train_y, val_y, test_y, train_X, val_X, test_X, scaler_target):

    full_y = pd.concat([train_y, val_y, test_y])
    full_X = pd.concat([train_X, val_X, test_X])

    model_test = model.apply(full_y, exog = full_X)
    preds_scaled = model_test.fittedvalues[test_y.index[0]:]

    preds_eval = scaler_target.inverse_transform(preds_scaled.values.reshape(-1, 1)).flatten()
    real_eval  = scaler_target.inverse_transform(test_y.values.reshape(-1, 1)).flatten()

    rmse = np.sqrt(mean_squared_error(real_eval, preds_eval))

    print("RMSE Best SARIMAX Model:", round(rmse, 2))

    # Image Real vs Predictions

    plt.figure(figsize = (18, 12))

    plt.plot(test_y.index, real_eval,label = "Real (Test)", color = "black", alpha = 0.7)
    plt.plot(test_y.index, preds_eval, label = f"Prediction SARIMAX", color = "seagreen")
    plt.title("Prediction vs Real — SARIMAX")
    plt.xlabel("Date")
    plt.ylabel("SPY Close Price")
    plt.grid(True, alpha = 0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(eval_pred_sarimax_path / "SARIMAX_test_prediction.png")
    plt.close()


# 5. Future prediction

def future_prediction(model, train_y, val_y, test_y, train_X, val_X, test_X, scaler_target, horizon = 30):

    full_y = pd.concat([train_y, val_y, test_y])
    full_X = pd.concat([train_X, val_X, test_X])

    final_model_fit = SARIMAX(
        endog = full_y,
        exog = full_X,
        order = model.model.order,
        seasonal_order = model.model.seasonal_order,
        enforce_stationarity = False,
        enforce_invertibility = False).fit(disp = False)

    # For future exogenous variables we use the lastest known value

    last_X = full_X.tail(1).values
    future_X = pd.DataFrame(np.repeat(last_X, horizon, axis = 0), columns = full_X.columns)

    preds_future_scaled = final_model_fit.forecast(steps = horizon, exog = future_X)
    preds_future_val = scaler_target.inverse_transform(preds_future_scaled.values.reshape(-1, 1)).flatten()

    last_date = full_y.index[-1]
    future_dates = pd.bdate_range(start = last_date + pd.Timedelta(days = 1), periods = horizon)

    df_future = pd.DataFrame({"Date":future_dates, "Predicted_Close":preds_future_val})
    df_future.set_index("Date", inplace = True)

    df_future.to_csv(eval_pred_sarimax_path / "future_30d_prediction_SARIMAX.csv")

    print("Future prediction saved")

    # Plot last 90 days + future

    plt.figure(figsize = (22, 20))

    recent_history = full_y.tail(90).values.reshape(-1, 1)
    recent_history_usd = scaler_target.inverse_transform(recent_history).flatten()
    recent_dates = full_y.tail(90).index

    plt.plot(recent_dates, recent_history_usd,label = "Recent History", color = "black")
    plt.plot(future_dates, preds_future_val, label = f"Prediction SARIMAX +{horizon}d", color = "seagreen", marker = "o")
    plt.title("SPY Close Price — Future Projection")
    plt.xlabel("Date")
    plt.ylabel("SPY Close Price")
    plt.grid(True, alpha = 0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(eval_pred_sarimax_path / "future_30d_projection_SARIMAX.png")
    plt.close()



# 6. Manual SARIMAX models evaluation + future forecast

def evaluate_manual_models(train_y, val_y, test_y,
                           train_X, val_X, test_X,
                           scaler_target,
                           horizon = 30):

    print("Evaluating manual SARIMAX models...")

    full_y = pd.concat([train_y, val_y, test_y])
    full_X = pd.concat([train_X, val_X, test_X])

    manual_models = {
        "Manual_1_(2,1,2)x(1,1,1,5)": ((2, 1, 2), (1, 1, 1, 5)),
        "Manual_2_(1,1,1)x(1,1,1,5)": ((1, 1, 1), (1, 1, 1, 5)),
        "Manual_3_(1,1,0)x(1,0,0,5)": ((1, 1, 0), (1, 0, 0, 5)),
    }

    predictions_30d = {}
    rmse_results = {}

    last_X = full_X.tail(1).values
    future_X = pd.DataFrame(np.repeat(last_X, horizon, axis = 0), columns = full_X.columns)

    last_date = full_y.index[-1]
    future_dates = pd.bdate_range(start = last_date + pd.Timedelta(days = 1), periods = horizon)

    # Last 90 days (real)

    recent_y = full_y.tail(90).values.reshape(-1, 1)
    recent_y_usd = scaler_target.inverse_transform(recent_y).flatten()
    recent_dates = full_y.tail(90).index

    for name, (order, seasonal_order) in manual_models.items():

        print(f"...  {name}")

        model = SARIMAX(
            endog = train_y,
            exog = train_X,
            order = order,
            seasonal_order = seasonal_order,
            enforce_stationarity = False,
            enforce_invertibility = False).fit(disp = False)

        # Test evaluation 

        model_test = model.apply(full_y, exog = full_X)
        preds_scaled = model_test.fittedvalues.loc[test_y.index]

        df_eval = pd.DataFrame({
            "real": test_y,
            "pred": preds_scaled
        }).dropna()

        if df_eval.empty:
            print("No valid test predictions (all NaNs) → model skipped")
            continue

        preds_eval = scaler_target.inverse_transform(
            df_eval["pred"].values.reshape(-1, 1)
        ).flatten()

        real_eval = scaler_target.inverse_transform(
            df_eval["real"].values.reshape(-1, 1)
        ).flatten()

        rmse = np.sqrt(mean_squared_error(real_eval, preds_eval))
        rmse_results[name] = rmse

        print(f"   RMSE: {rmse:.2f}")

        # Refit on full data for future forecast

        final_fit = SARIMAX(
            endog = full_y,
            exog = full_X,
            order = order,
            seasonal_order = seasonal_order,
            enforce_stationarity = False,
            enforce_invertibility = False).fit(disp = False)

        preds_future_scaled = final_fit.forecast(steps = horizon, exog = future_X)

        preds_future = scaler_target.inverse_transform(preds_future_scaled.values.reshape(-1, 1)).flatten()

        predictions_30d[name] = preds_future

        # Save CSV

        df_future = pd.DataFrame({
            "Date": future_dates,
            "Predicted_Close": preds_future
        }).set_index("Date")

        df_future.to_csv(
            eval_pred_sarimax_path / f"future_30d_{name}.csv"
        )

    # Combined Plot 

    plt.figure(figsize = (22, 20))

    plt.plot(
        recent_dates,
        recent_y_usd,
        color = "black",
        label = "Real (Last 90 days)",
        linewidth = 2
    )

    colors = ["palegreen", "deepskyblue", "mediumslateblue"]

    for (name, preds), color in zip(predictions_30d.items(), colors):

        plt.plot(
            future_dates,
            preds,
            marker = "o",
            linestyle = "--",
            color = color,
            label = f"{name} | RMSE = {rmse_results[name]:.2f}"
        )

    plt.title("SARIMAX Manual Models — 30-Day Projection")
    plt.xlabel("Date")
    plt.ylabel("SPY Close Price")
    plt.grid(True, alpha = 0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        eval_pred_sarimax_path / "future_30d_projection_MANUAL_MODELS.png"
    )
    plt.close()

    print("Manual models evaluation finished")



# ------------------------------------------------------------------------------------------------------------


# Main

if __name__ == '__main__':

    print("SARIMAX Model Pipeline...")
    
    # 1. Load data

    train_y, val_y, test_y, train_X, val_X, test_X, scaler_target = load_data()

    # 2. Optimal SARIMAX hyperparameters

    best_p, best_d, best_q, best_P, best_D, best_Q, s = search_best_sarimax_hyperparms(train_y, val_y, train_X, val_X)

    # 3. Train model

    model = train_model(train_y, train_X, best_p, best_d, best_q, best_P, best_D, best_Q, s)
    
    # 4. Test evaluation

    evaluate_test(model, train_y, val_y, test_y, train_X, val_X, test_X, scaler_target)

    # 5. Future prediction

    future_prediction(model, train_y, val_y, test_y, train_X, val_X, test_X, scaler_target, horizon = 30)

    # 6. Testing manual SARIMAX models  

    evaluate_manual_models(train_y, val_y, test_y, train_X, val_X, test_X, scaler_target, horizon = 30)  

    print("Pipeline done")