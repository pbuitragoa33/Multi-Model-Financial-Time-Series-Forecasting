# GBM Model (Geometric Brownian Motion)


# Data preparation --> work with log-returns to estimate tge parameters
# Parameters estimation (μ: Drift and σ: Volatility)
# Data preparation for testing --> train/test data
# Montecarlo Simulation for SPY testing a lot of trajectories (using Ito's correction)
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
eval_pred_gbm_path = os.getenv('OUT_EVAL_PRED_12_GBM')  

procesed_data_path = Path(procesed_data_path)
out_diverse_path = Path(out_diverse_path)
eval_pred_gbm_path = Path(eval_pred_gbm_path)



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


# 2. Parameter Estimations and KS Test according to data

# Parameter Estimation with Train set
# Display annualized parameter values (for informational purposes only)
# Kolmogorov–Smirnov Test (KS Test) ---> to assess whether the log returns of the S&P 500 follow a normal distribution, as assumed by the GBM

# H₀ (null hypothesis): log-normal returns follow a Normal(μ, σ)
# H₁: returns do not follow a normal distribution
# If H₀ get rejected, the GBM does not accurately describe the actual data


def params_calibration(train):

    # Historical Drift Estimation

    mu = np.mean(train)
    annualized_mu = mu * 252 

    print("Annualized Drift or Average Growth Rate: ", round(annualized_mu, 3))  

    # Historical Volatility Estimation

    sigma = np.std(train)
    annualized_sigma = sigma * np.sqrt(252)

    print("Annualized Volatility: ", round(annualized_sigma, 3))


    # Kolmogorov–Smirnov Test (KS Test) 

    # Scale

    z = (train - mu) / sigma

    ks_stat, p_value = stats.kstest(z, 'norm')

    print("KS statistic:", ks_stat)
    print("p-value:", p_value)

    alpha = 0.05

    if p_value >= alpha:

        print("The null hypothesis is not rejected (the data follow a normal distribution)")

    else:

        print("The null hypothesis is rejected (the data does not follow a normal distribution)")


    print("--" * 30)
    print("Drift - Average Growth Rate (μ): ", round(mu, 6))
    print("Volatility - Uncertainty (σ): ", round(sigma, 6))

    return mu, sigma


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


# 4. GBM Montecarlo Simulation

# Generates n_sims price trajectories for n_days and returns a matrix of shape (n_days, n_sims)

def simulation_gbm_paths(S0, mu, sigma, n_days, n_sims = 10000):

    # Dailt Step (t + 1)

    dt = 1

    # Initialize matrix

    paths = np.zeros((n_days, n_sims))

    paths[0] = S0

    # Ito's Correction for Drift for the geometric simulation ---> drift = (mu - 0.5 * sigma^2) * dt

    drift = (mu - 0.5 * (sigma ** 2)) * dt
    volatility = sigma * np.sqrt(dt)

    # Random Shocks (Brownian Motion) with Normal Distribution --> Z ~ N(0, 1)

    Z = np.random.normal(0, 1, (n_days - 1, n_sims))

    # Vectorized path calculation --> S_t = S_{t-1} * exp(drift + vol * Z)

    for t in range(1, n_days):

        paths[t] = paths[t - 1] * np.exp(drift + volatility * Z[t - 1])

    
    return paths


# 5. Evaluation using the test dataset (close price)

def evaluate_gbm(test_data, S0, mu, sigma, n_sims = 10000):

    print("Evaluating the simulation in contrast with test dataset, it contains: ", len(test_data), " days...")

    n_days = len(test_data)

    # Execute simulation using the previous function (simulation_gbm_paths)

    paths = simulation_gbm_paths(S0, mu, sigma, n_days, n_sims)

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
    plt.semilogy(mean_path, color = 'red', label = 'Mean Path GBM', linewidth = 3)
    plt.semilogy(median_path, color = 'blue', label = 'Median Path GBM', linewidth = 3)

    plt.title('GBM SPY Close Price Simulation', fontsize = 25)
    plt.xlabel('Days')
    plt.ylabel('Price')
    plt.grid(True)
    plt.legend(fontsize = 18)
    plt.savefig(out_diverse_path / '12_gbm_trajectories.png')
    plt.close()
    
    print("Visualization #1 Saved")


    # Visualization # 2

    plt.figure(figsize = (30, 22))

    plt.semilogy(test_data.index, test_data.values, color = 'black', label = 'Real SPY', linewidth = 2)
    plt.semilogy(test_data.index, mean_path, color = 'royalblue', linestyle = '--', label = 'GBM Mean Path', alpha = 0.7)
    plt.semilogy(test_data.index, median_path, color = 'orangered', linestyle = '--', label = 'GBM Median Path', alpha = 0.7)

    # Probability Cone

    plt.fill_between(test_data.index, lower_bound, upper_bound, color = 'silver', alpha = 0.3, label = '90% Confidence Interval')

    plt.title(f'GBM Validation: Simulation vs Real Prices (n = {n_sims})')
    plt.ylabel('SPY Close Price')
    plt.legend()
    plt.grid(True, alpha = 0.3)
    plt.tight_layout()
    plt.savefig(eval_pred_gbm_path / 'gbm_test_validation.png')
    plt.close()

    print("Visualization #2 Saved")


# 6. Future Prediction (30 day horizon)

def future_prediction(last_price, last_date, mu, sigma, horizon = 30):

    print("Doing 30-day forecast...")

    n_sims = 10000

    # Execute simulation using the previous function (simulation_gbm_paths)
    # + 1 to include the 0 day but this will be removed from the results

    paths = simulation_gbm_paths(last_price, mu, sigma, horizon + 1, n_sims)

    future_paths = paths[1:]

    # Metrics

    mean_forecast = np.mean(future_paths, axis = 1)
    median_forecast = np.median(future_paths, axis = 1)

    # Future dates creation

    future_dates = pd.bdate_range(start = last_date + pd.Timedelta(days = 1), periods = horizon)

    # Save as csv file

    df = pd.DataFrame({'Date': future_dates, 'GBM Mean': mean_forecast, 'GBM Median': median_forecast})
    df = df.set_index('Date')
    df.to_csv(eval_pred_gbm_path / 'gbm_future_prediction.csv')

    print("Prediction file saved...")

    # Visualization # 3 - Fan Chart

    plt.figure(figsize = (18, 12))

    # With median forecast

    plt.semilogy(future_dates, median_forecast, 'r-o', markersize = 4, label = 'GBM Median Projection')

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
    
    plt.title(f'Stochastic Model GBM Forecast for Next {horizon} days')
    plt.ylabel('SPY Close Price')
    plt.legend()
    plt.grid(True, alpha = 0.3)
    plt.tight_layout()
    plt.savefig(eval_pred_gbm_path / 'gbm_future_forecast.png')
    plt.close()

    print("Fan chart saved")




# --------------------------------------------------------------------------------------------------------------


# Main

if __name__ == "__main__":

    print("Pipeline Geometric Brownian Motion (GBM)...")

    # 1. Data preparation for SPY log-returns

    train_lgreturns, test_lgreturns = data_preparation_lgreturns() 

    # 2. Parameter Estimations and KS Test according to log returns data

    mu, sigma = params_calibration(train_lgreturns)

    # 3. Data preparation for SPY Close Price

    train_price, test_price = data_preparation_price()

    # 4. GBM Montecarlo Simulation

    #  The function will be invoked in the evaluation module

    #  Important to get the last price (S0 parameter --> Last train price)

    last_train_price = train_price.values[-1]

    # 5. GBM Evaluation 

    evaluate_gbm(test_price, last_train_price, mu, sigma, n_sims = 10000)

    # 6. Future Prediction

    last_test_price = test_price.values[-1]
    last_test_date = test_price.index[-1]

    future_prediction(last_test_price, last_test_date, mu, sigma, horizon = 30)

    print("GBM Pipeline Finished")