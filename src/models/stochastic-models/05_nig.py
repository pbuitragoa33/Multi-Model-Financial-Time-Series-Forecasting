# NIG Model (Normal Inverse Gaussian)


# Data preparation --> work with log-returns to estimate the parameters
# Parameters estimation (α: Tail Heaviness, β: Skewness, δ: Scale, μ: Location)
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
from scipy.stats import norminvgauss


# Path configurations

load_dotenv()

procesed_data_path = os.getenv('PROCESSED_DATA_PATH')  
out_diverse_path = os.getenv('OUT_DIVERSE_PATH')     
eval_pred_nig_path = os.getenv('OUT_EVAL_PRED_16_NIG')  

procesed_data_path = Path(procesed_data_path)
out_diverse_path = Path(out_diverse_path)
eval_pred_nig_path = Path(eval_pred_nig_path)



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


# 2. NIG Parameters Calibration (α, β, δ, μ)

def nig_params_estimation(train):

    # MLE will be used because of the library class import

    print("Calibrating NIG parameters via MLE...")

    alpha, beta, loc, scale = norminvgauss.fit(train.values)

    print("--" * 25)
    print(f"α (Tail Heaviness): {alpha:.4f}")
    print(f"β (Skewness):      {beta:.4f}")
    print(f"δ (Scale):         {scale:.6f}")
    print(f"μ (Location):      {loc:.6f}")

    return alpha, beta, scale, loc


# 3. Data preparation for SPY Close Price

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


# 4. NIG Montecarlo Simulation

def simulation_nig_paths(S0, alpha, beta, delta, mu, n_days, n_sims = 10000):

    # Dailt Step (t + 1)

    dt = 1

    # Initialize matrix

    paths = np.zeros((n_days, n_sims))

    paths[0] = S0

    # NIG Increments

    for t in range(1, n_days):

        X = norminvgauss.rvs(
            alpha, beta, 
            loc = mu * dt, scale = delta * np.sqrt(dt),
            size = n_sims
        )

        # Price dynamics --> St​ = S0 * ​exp(Xt​)

        paths[t] = paths[t - 1] * np.exp(X)

    return paths


# 5. Evaluation using the test dataset (close price)

def evaluate_nig(test_data, S0, alpha, beta, delta, mu, n_sims=10000):

    print("Evaluating NIG model...")

    n_days = len(test_data)

    paths = simulation_nig_paths(S0, alpha, beta, delta, mu, n_days, n_sims)

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
    plt.semilogy(mean_path, color = 'red', label = 'Mean Path NIG Model', linewidth = 3)
    plt.semilogy(median_path, color = 'blue', label = 'Median Path NIG Model', linewidth = 3)

    plt.title('NIG SPY Close Price Simulation', fontsize = 25)
    plt.xlabel('Days')
    plt.ylabel('Price')
    plt.grid(True)
    plt.legend(fontsize = 18)
    plt.savefig(out_diverse_path / '16_nig_trajectories.png')
    plt.close()
    
    print("Visualization #1 Saved")


    # Visualization # 2

    plt.figure(figsize = (30, 22))

    plt.semilogy(test_data.index, test_data.values, color = 'black', label = 'Real SPY', linewidth = 2)
    plt.semilogy(test_data.index, mean_path, color = 'royalblue', linestyle = '--', label = 'NIG Mean Path', alpha = 0.7)
    plt.semilogy(test_data.index, median_path, color = 'orangered', linestyle = '--', label = 'NIG Median Path', alpha = 0.7)

    # Probability Cone

    plt.fill_between(test_data.index, lower_bound, upper_bound, color = 'silver', alpha = 0.3, label = '90% Confidence Interval')

    plt.title(f'NIG Model Validation: Simulation vs Real Prices (n = {n_sims})')
    plt.ylabel('SPY Close Price')
    plt.legend()
    plt.grid(True, alpha = 0.3)
    plt.tight_layout()
    plt.savefig(eval_pred_nig_path / 'nig_test_validation.png')
    plt.close()

    print("Visualization #2 Saved")


# 6. Future Preidction (30-day forecast)

def future_prediction(last_price, last_date, alpha, beta, delta, mu, horizon = 30):

    print("Running NIG prediction...")

    paths = simulation_nig_paths(last_price, alpha, beta, delta, mu, horizon + 1)

    future_paths = paths[1:]

   # Metrics

    mean_forecast = np.mean(future_paths, axis = 1)
    median_forecast = np.median(future_paths, axis = 1)

    # Future dates creation

    future_dates = pd.bdate_range(start = last_date + pd.Timedelta(days = 1), periods = horizon)

    # Save as csv file

    df = pd.DataFrame({'Date': future_dates, 'NIG Mean': mean_forecast, 'NIG Median': median_forecast})
    df = df.set_index('Date')
    df.to_csv(eval_pred_nig_path / 'nig_future_prediction.csv')

    print("Prediction file saved...")

   # Visualization # 3 - Fan Chart

    plt.figure(figsize = (18, 12))

    # With median forecast

    plt.semilogy(future_dates, median_forecast, 'r-o', markersize = 4, label = 'NIG Median Projection')

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
    
    plt.title(f'Stochastic NIG (Normal Inverse Gaussian) Model Forecast for Next {horizon} days')
    plt.ylabel('SPY Close Price')
    plt.legend()
    plt.grid(True, alpha = 0.3)
    plt.tight_layout()
    plt.savefig(eval_pred_nig_path / 'nig_future_forecast.png')
    plt.close()

    print("Fan chart saved")




# --------------------------------------------------------------------------------------------------------------


# Main

if __name__ == "__main__":

    print("Pipeline Normal Inverse Gaussian (NIG)...")

    # 1. Data preparation for SPY log-returns

    train_lgreturns, test_lgreturns = data_preparation_lgreturns() 

    # 2. NIG Parameters Estiamtion

    alpha, beta, delta, mu = nig_params_estimation(train_lgreturns)

    # 3. Data preparation for SPY Close Price

    train_price, test_price = data_preparation_price()

    # 4. NIG Model Montecarlo Simulation

    # 5. NIG Evaluation

    last_train_price = train_price.values[-1]

    evaluate_nig(test_price, last_train_price, alpha, beta, delta, mu)

    # 6. NIG model forecast

    last_test_price = test_price.values[-1]
    last_test_date = test_price.index[-1]

    future_prediction(last_test_price, last_test_date, alpha, beta, delta, mu, horizon = 30)

    print("NIG Pipeline Finished")