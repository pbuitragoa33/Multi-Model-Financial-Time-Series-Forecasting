# GARCH Model


# Data preparation for variance treatment
# Hyperparameter tuning (p, q values and distribution type)
# Best model training
# Conditional volatility plot
# 30 days volatility forecast


# Necessary libraries

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import joblib
from pathlib import Path
from arch import arch_model


# Paths configuration

load_dotenv()

procesed_data_path = os.getenv('PROCESSED_DATA_PATH')

out_diverse_path = os.getenv('OUT_DIVERSE_PATH')      # Diverse savable items
out_model_path = os.getenv('OUT_MODEL_PATH_CLASSIC')  # Save the model as joblib

eval_pred_garch_path = os.getenv('OUT_EVAL_PRED_10_GARCH')  # Save the prediction (csv and image)

procesed_data_path = Path(procesed_data_path)
out_diverse_path = Path(out_diverse_path)
out_model_path = Path(out_model_path)
eval_pred_garch_path = Path(eval_pred_garch_path)



# -------------------------------------- Functions --------------------------------------


# 1. Data preparation

def data_preparation():

    print("Loading log returns dataset...")

    df = pd.read_csv(procesed_data_path / 'spy_daily_log_returns.csv', index_col = 0, parse_dates = True)
    df = pd.DataFrame(df)

    # Turn into businnes days  (log_returns column)

    df = df['SPY_Returns'].asfreq('B').ffill().bfill()

    # Sort the dataframe by date (checkup)

    df = df.sort_index()

    # Define cuts (80% for train, 10% for validation, 10% for evaluation)

    n = len(df)
    train_end = int(n * 0.8)
    validation_end = int(n * 0.9)

    train = df.iloc[:train_end]
    validation = df.iloc[train_end:validation_end]
    test = df.iloc[validation_end:]

    # Re-scaling log-returns due to optimizer warning

    train = train * 100
    validation = validation * 100
    test = test * 100

    print(f"    - Train (80%): {train.shape[0]} trading days")
    print(f"    - Validation (10%):   {validation.shape[0]} trading days")
    print(f"    - Testing (10%):  {test.shape[0]} trading days")

    return train, validation, test


# 2. Hyperparameter search (p, q, distribution)

def search_best_garch(train, p_max = 5, q_max = 5):

    print("Searching best GARCH(p, q) and distribution...")

    results = []

    distributions = {"normal": "normal", "student_t": "t"}

    for p in range(1, p_max + 1):

        for q in range(1, q_max + 1):

            for dist_name, dist_code in distributions.items():

                model = arch_model(train,
                    mean = "Zero",
                    vol = "GARCH",
                    p = p,
                    q = q,
                    dist = dist_code)

                fit = model.fit(disp = "off")

                results.append({
                    "p": p,
                    "q": q,
                    "distribution": dist_name,
                    "aic": fit.aic,
                    "bic": fit.bic,
                    "loglik": fit.loglikelihood
                    })

    results_df = pd.DataFrame(results).sort_values("aic")

    best_row = results_df.iloc[0]

    print(
        f"Best model --> GARCH({int(best_row.p)}, {int(best_row.q)}), "
        f"dist={best_row.distribution}, "
        f"AIC={best_row.aic:.2f}"
    )

    results_df.to_csv(out_diverse_path / "10_garch_hyperparameter_search.csv", index = False)

    return int(best_row.p), int(best_row.q), best_row.distribution, results_df


# 3. Train final GARCH model

def train_garch_model(train, p, q, distribution):

    dist_code = "t" if distribution == "student_t" else "normal"

    model = arch_model(train,
        mean = "Zero",
        vol = "GARCH",
        p = p,
        q = q,
        dist = dist_code
    )

    model_fitted = model.fit(disp = "off")

    joblib.dump(model_fitted, out_model_path / "10_garch_model.joblib")

    print("Final GARCH model trained and saved.")

    return model_fitted


# 4. Conditional volatility plot

def plot_conditional_volatility(model_fitted):

    cond_vol = model_fitted.conditional_volatility

    plt.figure(figsize = (18, 16))

    plt.plot(cond_vol, color = "darkblue", label = "Conditional Volatility")
    plt.title("GARCH Conditional Volatility")
    plt.xlabel("Date")
    plt.ylabel("Volatility")
    plt.grid(alpha = 0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(eval_pred_garch_path / "GARCH_conditional_volatility.png")
    plt.close()


# 5. Volatility forecast

def forecast_volatility(model_fitted, last_date, horizon = 30):

    forecasts = model_fitted.forecast(horizon = horizon)

    variance_fc = forecasts.variance.iloc[-1]
    volatility_fc = np.sqrt(variance_fc)

    future_dates = pd.bdate_range(start = last_date + pd.Timedelta(days = 1), periods = horizon)

    vol_df = pd.DataFrame({"Forecasted_Volatility": volatility_fc.values}, index = future_dates)

    vol_df.to_csv(eval_pred_garch_path / "future_30d_volatility_GARCH.csv")

    print("Volatility forecast saved.")

    # Iamage: Volatility forecast

    plt.figure(figsize = (20, 18))

    plt.plot(vol_df.index, vol_df["Forecasted_Volatility"], marker = "o", color = "darkblue")
    plt.title("30-Day Volatility Forecast (GARCH)")
    plt.grid(alpha = 0.3)
    plt.tight_layout()

    plt.savefig(eval_pred_garch_path / "future_30d_volatility_GARCH.png")
    plt.close()



# ------------------------------------------------------------------------------------------------------------

# Main

if __name__ == "__main__":

    print("GARCH Pipeline (No Mean Model)...")

    # 1. Data preparation

    train, val, test = data_preparation()

    # 2. Hyperparameter tuning

    best_p, best_q, best_dist, search_results = search_best_garch(train, p_max = 5, q_max = 5)

    # 3. Final model training

    garch_fit = train_garch_model(train, best_p, best_q, best_dist)

    # 4. Conditional volatility

    plot_conditional_volatility(garch_fit)

    # 5. 30-day volatility forecast

    last_date = train.index[-1]

    forecast_volatility(garch_fit, last_date, horizon = 30)

    print("Pipeline finished successfully.")