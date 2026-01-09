# Heston Model


# Data preparation --> work with log-returns to estimate the parameters
# Parameters estimation (μ: Drift, κ: Speed of Mean Reversion, θ: Long-term Variance, X or σv: Volatility of Volatility, ρ: Correlation between Noises)
# Montecarlo Simulation for SPY testing a lot of trajectories
# Evaluation (RMSE and MAE)
# Prediction (30 days ahead)


# Necessary libraries 

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy import stats


# Path configurations

load_dotenv()

procesed_data_path = os.getenv('PROCESSED_DATA_PATH')  
out_diverse_path = os.getenv('OUT_DIVERSE_PATH')     
eval_pred_heston_path = os.getenv('OUT_EVAL_PRED_14_HESTON')  

procesed_data_path = Path(procesed_data_path)
out_diverse_path = Path(out_diverse_path)
eval_pred_heston_path = Path(eval_pred_heston_path)



# -------------------------------------- Functions --------------------------------------

# 1. Data Preparation for SPY log-returns

def data_preparation_lgreturns():

    print("Loading SPY log-returns dataset...")

    df = pd.read_csv(procesed_data_path / 'spy_daily_log_returns.csv', index_col = 0, parse_dates = True)
    df = pd.DataFrame(df)

    # Turn into businnes days (log-returns column)

    df = df['SPY_Returns'].asfreq('B').ffill().bfill()

    # Sort the dataframe by date (checkup)

    df = df.sort_index()

    # Define cuts (90% for train and 10% for evaluation)

    n = len(df)
    train_len = int(n * 0.9)

    train_lgreturns = df.iloc[:train_len]
    test_lgreturns = df.iloc[train_len:]

    print(f"    - Train log-returns (90%): {train_lgreturns.shape[0]} trading days")
    print(f"    - Test log-returns (10%):   {test_lgreturns.shape[0]} trading days")

    return train_lgreturns, test_lgreturns


# 2. Drift Estimation (μ) from GBM 

def drift_calibration(train):

    mu = np.mean(train)

    print("Drift μ: ", round(mu, 6))

    return mu


# 3. Heston Model Parameter Calibration 

# Honestly, I didn't know how to approach this estimation, so the AI suggested the next heuristic

def heston_params_estimation(train):

    # Proxy for instant variance E[rt2​] ≈ vt * ​Δt

    instant_var = train ** 2

    # θ: Long-term variance --> The "normal" or structural or average level of volatility

    theta = np.mean(instant_var)

    # κ: Speed of reversion to the mean. Balanced value.

    kappa = 1.5

    # X or σv: Volatility of Volatility

    xi = np.std(instant_var)

    # ρ: Leverage correlation --> Correlation between return and future variance

    rho = np.corrcoef(train[: -1], instant_var[1:])[0, 1]

    print("Heston Parameters")
    print("--" * 25)
    print(f"kappa (κ): {kappa:.4f}")
    print(f"theta (θ): {theta:.6f}")
    print(f"xi (ξ or σv): {xi:.6f}")
    print(f"rho (ρ): {rho:.4f}")

    return kappa, theta, xi, rho


# 4. Data preparation for SPY Close Price

def data_preparation_price():

    print("Loading SPY close price dataset...")

    df = pd.read_csv(procesed_data_path / 'spy_daily_close.csv', index_col = 0, parse_dates = True)
    df = pd.DataFrame(df)

    # Turn into businnes days (Close Price column)

    df = df['Close'].asfreq('B').ffill().bfill()

    # Sort the dataframe by date (checkup)

    df = df.sort_index()

    # Define cuts (90% for train and 10% for evaluation)

    n = len(df)
    train_len = int(n * 0.9)

    train_price = df.iloc[:train_len]
    test_price = df.iloc[train_len:]

    print(f"    - Train close price (90%): {train_price.shape[0]} trading days")
    print(f"    - Test close price (10%):   {test_price.shape[0]} trading days")

    return train_price, test_price


# 5. Heston Model Montecarlo Simulation

def simulation_heston_paths(S0, mu, kappa, theta, xi, rho, v0, n_days, n_sims = 10000):

    # Dailt Step (t + 1)

    dt = 1

    # Initialize matrices (S: price, v: stochastic variance)

    S = np.zeros((n_days, n_sims))  
    v = np.zeros((n_days, n_sims))

    # Initial conditions

    S[0] = S0
    v[0] = v0

    # Correlated Brownian Noises (Z1 anf Z2) 
    # Price noise and Volatility noise and correlation between them

    Z1 = np.random.normal(size = (n_days - 1, n_sims))
    Z2 = rho * Z1 + np.sqrt(1 - rho ** 2) * np.random.normal(size = (n_days - 1, n_sims))

    for t in range(1, n_days):

        # Negative variances must be avoided

        v_prev = np.maximum(v[t - 1], 0)

        # Variance dynamic - CIR Process (Mean reversion process)

        v[t] = (
            v_prev+ kappa * (theta - v_prev) * dt + xi * np.sqrt(v_prev * dt) * Z2[t - 1]
                )
        
        v[t] = np.maximum(v[t], 0)

        # Price dynamic (similar respect GBM but assumes stochastic volatiloity)

        S[t] = S[t - 1] * np.exp((mu - 0.5 * v_prev) * dt+ np.sqrt(v_prev * dt) * Z1[t - 1])

    return S, v
    

# 6. Evaluation using the test dataset (close price)

def evaluate_heston(test_data, S0, mu, kappa, theta, xi, rho, v0, n_sims = 10000):

    print("Evaluating Heston model...")

    n_days = len(test_data)

    paths, vols = simulation_heston_paths(S0, mu, kappa, theta, xi, rho, v0, n_days, n_sims)

    # Mean and Median calculation

    mean_path = np.mean(paths, axis = 1)
    median_path = np.median(paths, axis = 1)

    # Confidence Intervals (5% and 95%)

    lower_bound = np.percentile(paths, 5, axis = 1)
    upper_bound = np.percentile(paths, 95, axis = 1)

    # Metrics Calculation (RMSE and MAE in contrast with mean and median measures)

    rmse_mean = np.sqrt(mean_squared_error(test_data.values, mean_path))
    rmse_median = np.sqrt(mean_squared_error(test_data.values, median_path))

    mae_mean = mean_absolute_error(test_data.values, mean_path)
    mae_median = mean_absolute_error(test_data.values, median_path)

    print(f"   - RMSE (Mean Path):   $ {rmse_mean:.2f}")
    print(f"   - RMSE (Median Path): $ {rmse_median:.2f}")
    print(f"   - MAE (Mean Path):   $ {mae_mean:.2f}")
    print(f"   - MAE (Median Path): $ {mae_median:.2f}")

   # Visualization # 1

    plt.figure(figsize = (18, 12))

    plt.semilogy(paths, color = 'gray', alpha = 0.3)
    plt.semilogy(mean_path, color = 'red', label = 'Mean Path Heston Model', linewidth = 3)
    plt.semilogy(median_path, color = 'blue', label = 'Median Path Heston Model', linewidth = 3)

    plt.title('Heston SPY Close Price Simulation', fontsize = 25)
    plt.xlabel('Days')
    plt.ylabel('Price')
    plt.grid(True)
    plt.legend(fontsize = 18)
    plt.savefig(out_diverse_path / '14_heston_trajectories.png')
    plt.close()
    
    print("Visualization #1 Saved")


    # Visualization # 2

    plt.figure(figsize = (30, 22))

    plt.semilogy(test_data.index, test_data.values, color = 'black', label = 'Real SPY', linewidth = 2)
    plt.semilogy(test_data.index, mean_path, color = 'royalblue', linestyle = '--', label = 'Heston Mean Path', alpha = 0.7)
    plt.semilogy(test_data.index, median_path, color = 'orangered', linestyle = '--', label = 'Heston Median Path', alpha = 0.7)

    # Probability Cone

    plt.fill_between(test_data.index, lower_bound, upper_bound, color = 'silver', alpha = 0.3, label = '90% Confidence Interval')

    plt.title(f'Heston Model Validation: Simulation vs Real Prices (n = {n_sims})')
    plt.ylabel('SPY Close Price')
    plt.legend()
    plt.grid(True, alpha = 0.3)
    plt.tight_layout()
    plt.savefig(eval_pred_heston_path / 'heston_test_validation.png')
    plt.close()

    print("Visualization #2 Saved")


# 7. Future Prediction (30 day horizon)

def future_prediction(last_price, last_date, mu, kappa, theta, xi, rho, v0, horizon = 30):

    print("Running prediction...")

    paths, vols = simulation_heston_paths(last_price, mu, kappa, theta, xi, rho, v0, horizon + 1)

    future_paths = paths[1:]

    # Metrics

    mean_forecast = np.mean(future_paths, axis = 1)
    median_forecast = np.median(future_paths, axis = 1)

    # Future dates creation

    future_dates = pd.bdate_range(start = last_date + pd.Timedelta(days = 1), periods = horizon)

    # Save as csv file

    df = pd.DataFrame({'Date': future_dates, 'Heston Mean': mean_forecast, 'Heston Median': median_forecast})
    df = df.set_index('Date')
    df.to_csv(eval_pred_heston_path / 'heston_future_prediction.csv')

    print("Prediction file saved...")

    # Visualization # 3 - Fan Chart

    plt.figure(figsize = (18, 12))

    # With median forecast

    plt.semilogy(future_dates, median_forecast, 'r-o', markersize = 4, label = 'Heston Median Projection')

    # Probability Cones - Fan Chart

    # Very probable zone (25% - 75%)

    plt.fill_between(future_dates, 
                     np.percentile(future_paths, 25, axis = 1),
                     np.percentile(future_paths, 75, axis = 1),
                     color = 'red', alpha = 0.35, label = '50% Probability')
    
    # Very extreme zone (5% and 95%)

    plt.fill_between(future_dates, 
                     np.percentile(future_paths, 5, axis = 1),
                     np.percentile(future_paths, 95, axis = 1),
                     color = 'red', alpha = 0.15, label = '90% Probability')
    
    plt.title(f'Stochastic Heston Model Forecast for Next {horizon} days')
    plt.ylabel('SPY Close Price')
    plt.legend()
    plt.grid(True, alpha = 0.3)
    plt.tight_layout()
    plt.savefig(eval_pred_heston_path / 'heston_future_forecast.png')
    plt.close()

    print("Fan chart saved")




# --------------------------------------------------------------------------------------------------------------


# Main

if __name__ == "__main__":

    print("Pipeline Heston Model...")

    # 1. Data preparation for SPY log-returns

    train_lgreturns, test_lgreturns = data_preparation_lgreturns()

    # 2. Parameter Estimations (Drift)

    mu = drift_calibration(train_lgreturns)

    # 3. Heston Parameters "Estimation"

    kappa, theta, xi, rho = heston_params_estimation(train_lgreturns)

    # 4. Data preparation for SPY Close Price

    train_price, test_price = data_preparation_price()

    # 5. Heston Model Montecarlo Simulation

    # 6. Heston Evaluation

    v0 = np.var(train_lgreturns)
    last_train_price = train_price.values[-1]

    evaluate_heston(test_price, last_train_price, mu, kappa, theta, xi, rho, v0, n_sims = 10000)

    # 7. Future Prediction

    last_test_price = test_price.values[-1]
    last_test_date = test_price.index[-1]

    future_prediction(last_test_price, last_test_date, mu, kappa, theta, xi, rho, v0, horizon = 30)

    print("Heston Pipeline Finished")