# Kou Model (Double Exponential Jump Diffusion)


# Data preparation --> work with log-returns to estimate the parameters
# Parameters estimation (μ: Drift and σ: Volatility, λ: Number of expected jumps, η1 and η2: Profit/Losses decay velocity, p: Positive jump probability)
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
eval_pred_kou_path = os.getenv('OUT_EVAL_PRED_15_KOU')  

procesed_data_path = Path(procesed_data_path)
out_diverse_path = Path(out_diverse_path)
eval_pred_kou_path = Path(eval_pred_kou_path)



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



# 2. Drift and Volatility Estimation (from GBM component)

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


# 3. Kou Model Parameters Calibration (λ, p, η1, η2)

def kou_params_estimation(train, z_threshold = 3):

    # Extreme events identification (jumps where the absolute value > 3 std)
    # As you can see in the PDF, the extremes are the jumps that the GBM can't explain

    # Identify extreme returns as jumps (> 3σ)

    jumps = train[np.abs(train) > z_threshold * np.std(train)]

    # # λ is the intensity of jumps (Poisson Process). Is the daily probability of a jump occurring 

    lambda_jump = len(jumps) / len(train)

    # Identify positive and neagtive jumps for eta estiamtion

    positive_jumps = jumps[jumps > 0]
    negative_jumps = jumps[jumps < 0]

    # Probability of positve jumps

    p = len(positive_jumps) / len(negative_jumps) if len(jumps) > 0 else 0.5

    # Avoid problematic values for η1 and η2

    eta1_r = 1 / (np.mean(positive_jumps) + 1e-8) if len(positive_jumps) > 0 else 1.5
    eta1 = max(1.01, eta1_r)

    eta2_r = -1 / (np.mean(negative_jumps) + 1e-8) if len(negative_jumps) > 0 else 1.5
    eta2 = max(1.01, eta2_r)

    print("--" * 25)
    print("Kou Jump parameters:")
    print(f"λ (lambda_jump): {lambda_jump:.6f}")
    print(f"p (probability positive jump): {p:.3f}")
    print(f"η1 (eta1, positive jump rate): {eta1:.3f}")
    print(f"η2 (eta2, negative jump rate): {eta2:.3f}")

    return lambda_jump, p, eta1, eta2


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


# 5. Kou Montecarlo Simulation

def simulation_kou_paths(S0, mu, sigma, lambda_jump, p, eta1, eta2,  n_days, n_sims = 10000):

    # Dailt Step (t + 1)

    dt = 1

    # Initialize matrix

    paths = np.zeros((n_days, n_sims))

    paths[0] = S0

    # Z --> Brownian Motion (Noise)
    # N --> Poisson Process (Jumps)
    # Y --> Double Exponential (Jump sizes)

    Z = np.random.normal(size=(n_days - 1, n_sims))

    N = np.random.poisson(lambda_jump * dt, size = (n_days - 1, n_sims))

    Y = np.zeros((n_days - 1, n_sims))

    for i in range(n_days - 1):

        U = np.random.uniform(size = n_sims)

        # Vectorized jump size per loop
        # If U < p: Positive jump --> Y ∼ Exp(η1​)
        # If U > p: Negative jump --> Y ∼ − Exp(η2​)

        jump_sizes = np.where(U < p, 
                             np.random.exponential(scale = 1 / eta1, size = n_sims), 
                             - np.random.exponential(scale = 1 / eta2, size = n_sims))
        
        Y[i] = jump_sizes

    # Drift correction --> κ = E[(e^Y) − 1] 
    # mu: Average return, -0.5 σ²: GBM term, -λκ: jump compensation

    kappa = p * eta1 / (eta1 - 1) + (1 - p) * eta2 / (eta2 + 1) - 1
    drift = (mu - 0.5 * sigma ** 2 - lambda_jump * kappa) * dt

    # Price dynamic
    # N = 0 --> No jump
    # N = 1 --> One jump
    # N > 1 --> Jump accumulation

    for t in range(1, n_days):

        jump_component = np.exp(Y[t - 1] * N[t - 1])

        paths[t] = paths[t - 1] * np.exp(drift + sigma * np.sqrt(dt) * Z[t - 1]) * jump_component

    return paths


# 6. Evaluation using the test dataset (close price)

def evaluate_kou(test_data, S0, mu, sigma, lambda_jump, p, eta1, eta2, n_sims = 10000):

    print("Evaluating Kou model...")

    n_days = len(test_data)

    paths = simulation_kou_paths(S0, mu, sigma, lambda_jump, p, eta1, eta2, n_days, n_sims)

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
    plt.semilogy(mean_path, color = 'red', label = 'Mean Path Kou Model', linewidth = 3)
    plt.semilogy(median_path, color = 'blue', label = 'Median Path Kou Model', linewidth = 3)

    plt.title('Kou SPY Close Price Simulation', fontsize = 25)
    plt.xlabel('Days')
    plt.ylabel('Price')
    plt.grid(True)
    plt.legend(fontsize = 18)
    plt.savefig(out_diverse_path / '15_kou_trajectories.png')
    plt.close()
    
    print("Visualization #1 Saved")


    # Visualization # 2

    plt.figure(figsize = (30, 22))

    plt.semilogy(test_data.index, test_data.values, color = 'black', label = 'Real SPY', linewidth = 2)
    plt.semilogy(test_data.index, mean_path, color = 'royalblue', linestyle = '--', label = 'Kou Mean Path', alpha = 0.7)
    plt.semilogy(test_data.index, median_path, color = 'orangered', linestyle = '--', label = 'Kou Median Path', alpha = 0.7)

    # Probability Cone

    plt.fill_between(test_data.index, lower_bound, upper_bound, color = 'silver', alpha = 0.3, label = '90% Confidence Interval')

    plt.title(f'Kou Model Validation: Simulation vs Real Prices (n = {n_sims})')
    plt.ylabel('SPY Close Price')
    plt.legend()
    plt.grid(True, alpha = 0.3)
    plt.tight_layout()
    plt.savefig(eval_pred_kou_path / 'kou_test_validation.png')
    plt.close()

    print("Visualization #2 Saved")


# 7. Future Prediction (30-day horizon)

def future_prediction(last_price, last_date, mu, sigma, lambda_jump, p, eta1, eta2, horizon = 30):

    print("Running prediction...")

    paths = simulation_kou_paths(last_price, mu, sigma, lambda_jump, p, eta1, eta2, horizon + 1)

    future_paths = paths[1:]

    # Metrics

    mean_forecast = np.mean(future_paths, axis = 1)
    median_forecast = np.median(future_paths, axis = 1)

    # Future dates creation

    future_dates = pd.bdate_range(start = last_date + pd.Timedelta(days = 1), periods = horizon)

    # Save as csv file

    df = pd.DataFrame({'Date': future_dates, 'Kou Mean': mean_forecast, 'Kou Median': median_forecast})
    df = df.set_index('Date')
    df.to_csv(eval_pred_kou_path / 'kou_future_prediction.csv')

    print("Prediction file saved...")

   # Visualization # 3 - Fan Chart

    plt.figure(figsize = (18, 12))

    # With median forecast

    plt.semilogy(future_dates, median_forecast, 'r-o', markersize = 4, label = 'Kou Median Projection')

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
    
    plt.title(f'Stochastic Kou (Double Exponential Jump Diffusion) Model Forecast for Next {horizon} days')
    plt.ylabel('SPY Close Price')
    plt.legend()
    plt.grid(True, alpha = 0.3)
    plt.tight_layout()
    plt.savefig(eval_pred_kou_path / 'kou_future_forecast.png')
    plt.close()

    print("Fan chart saved")




# --------------------------------------------------------------------------------------------------------------


# Main

if __name__ == "__main__":

    print("Pipeline Double Exponential Jump Diffusion (Kou)...")

    # 1. Data preparation for SPY log-returns

    train_lgreturns, test_lgreturns = data_preparation_lgreturns() 

    # 2. Parameter Estimations and KS Test according to log returns data (from GBM)

    mu, sigma = params_calibration(train_lgreturns)

    # 3. Kou Parameters Estimation

    lambda_jump, p, eta1, eta2 = kou_params_estimation(train_lgreturns)

    # 4. Data preparation for SPY Close Price

    train_price, test_price = data_preparation_price()

    # 5. Kou Model Montecarlo Simulation

    # 6. Kou Evaluation

    last_train_price = train_price.values[-1]

    evaluate_kou(test_price, last_train_price, mu, sigma, lambda_jump, p, eta1, eta2)

    # 7. Kou Model Forecast

    last_test_price = test_price.values[-1]
    last_test_date = test_price.index[-1]

    future_prediction(last_test_price, last_test_date, mu, sigma, lambda_jump, p, eta1, eta2, horizon = 30)

    print("Kou Pipeline Finished")