# DCC-GARCH Model (No Mean Model)


# Multivariate volatility modeling
# Individual GARCH(1,1) per asset
# Dynamic Conditional Correlation (DCC)
# Volatility & correlation forecasting


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

prepared_data_path = os.getenv('PREPARED_DATA_PATH')        # Load the data 
procesed_data_path = os.getenv('PROCESSED_DATA_PATH')

scalers_path = os.getenv('OUT_OBJECTS_PATH')  

out_diverse_path = os.getenv('OUT_DIVERSE_PATH')      # Diverse savable items
out_model_path = os.getenv('OUT_MODEL_PATH_CLASSIC')  # Save the model as joblib

eval_pred_dccgarch_path = os.getenv('OUT_EVAL_PRED_11_DCCGARCH')  # Save the prediction (csv and image)

prepared_data_path = Path(prepared_data_path)
scalers_path = Path(scalers_path)
procesed_data_path = Path(procesed_data_path)
out_diverse_path = Path(out_diverse_path)
out_model_path = Path(out_model_path)
eval_pred_dccgarch_path = Path(eval_pred_dccgarch_path)



# -------------------------------------- Functions --------------------------------------


# 1. Prepare data (load datasets, desnormalized dataframes, calculate log-returns, select most significant, spit and rescale)


CANDIDATE_TARGETS = {
    "Close_SPY": "SPY_ret",
    "gold": "gold_ret",
    "oil": "oil_ret",
    "tlt": "tlt_ret",
    "dxy": "dxy_ret",
    "vix": "vix_ret"
}


def load_and_prepare_multivariate_log_returns(top_k = 4):

    print("Loading normalized datasets and scalers...")

    # Load datasets

    train_df = pd.read_csv(prepared_data_path / "train_dataset.csv", index_col = 0, parse_dates = True)

    val_df = pd.read_csv(prepared_data_path / "validation_dataset.csv", index_col = 0, parse_dates = True)

    test_df = pd.read_csv(prepared_data_path / "test_dataset.csv", index_col = 0, parse_dates = True)

    # Load scalers

    target_scaler = joblib.load(scalers_path / "target_robust_scaler.joblib")
    features_scaler = joblib.load(scalers_path / "features_robust_scaler.joblib")

    # Candidate columns

    candidate_cols = list(CANDIDATE_TARGETS.keys())

    # Concatenate full dataset

    full_scaled = pd.concat([train_df[candidate_cols], val_df[candidate_cols], test_df[candidate_cols]])

    # Business days

    full_scaled = (full_scaled.asfreq("B").ffill().bfill().sort_index())

    # 1. Desnormalize each column

    full_prices = pd.DataFrame(index = full_scaled.index)

    for col in candidate_cols:

        if col == "Close_SPY":

            scaler = target_scaler

            full_prices[col] = scaler.inverse_transform(full_scaled[[col]]).flatten()

        else:

            scaler = features_scaler

            idx = list(train_df.columns).index(col)

            full_prices[col] = full_scaled[col] * scaler.scale_[idx] + scaler.center_[idx]

    # 2. Calculate log-returns

    log_returns = np.log(full_prices).diff().dropna()

    log_returns.columns = [CANDIDATE_TARGETS[c] for c in log_returns.columns]

    # 3. Select most significant features as target (DCC-GARCH allows multiple targey)
 
    if top_k is not None:

        vol_score = log_returns.std()
        corr_score = log_returns.corr().abs().mean()
        significance_score = vol_score * corr_score

        selected_cols = (significance_score.sort_values(ascending = False).head(top_k).index.tolist())

        log_returns = log_returns[selected_cols]

        print("Selected DCC targets:", selected_cols)

    else:

        selected_cols = log_returns.columns.tolist()

    # 4. Split (80% for training / 10% for validation / 10% for testing)

    n = len(log_returns)
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)

    train = log_returns.iloc[:train_end]
    validation = log_returns.iloc[train_end:val_end]
    test = log_returns.iloc[val_end:]

    # 5. Rescale (multiplying by 100)

    train *= 100
    validation *= 100
    test *= 100

    print(f"    - Train: {train.shape}")
    print(f"    - Validation: {validation.shape}")
    print(f"    - Test: {test.shape}")

    return train, validation, test, selected_cols


# 2. Univariate GARCH and residuals

def fit_univariate_garch(train, p = 1, q = 1, distribution = "student_t"):

    dist_code = "t" if distribution == "student_t" else "normal"

    results = {}
    sigmas = []
    std_resids = []

    for col in train.columns:

        model = arch_model(train[col],
            mean = "Zero",
            vol = "GARCH",
            p = p,
            q = q,
            dist = dist_code
        )

        res = model.fit(disp = "off")

        results[col] = res
        sigmas.append(res.conditional_volatility.values)
        std_resids.append(res.std_resid.values)

    sigmas = np.column_stack(sigmas)
    Z = np.column_stack(std_resids)

    return results, sigmas, Z


# 3. Recursive DCC 

def estimate_dcc(Z, a = 0.05, b = 0.93):

    T, N = Z.shape
    Qbar = np.cov(Z.T)

    Qt = np.zeros((T, N, N))
    Rt = np.zeros((T, N, N))

    Qt[0] = Qbar

    for t in range(1, T):

        Qt[t] = (
            (1 - a - b) * Qbar +
            a * np.outer(Z[t - 1], Z[t - 1]) +
            b * Qt[t - 1]
        )

        D_inv = np.diag(1.0 / np.sqrt(np.diag(Qt[t])))
        Rt[t] = np.dot(np.dot(D_inv, Qt[t]), D_inv)


    return Rt, Qt


# 4. Conditional Covariance Matrix

def compute_conditional_covariance(sigmas, Rt):

    T, N = sigmas.shape
    Ht = np.zeros((T, N, N))

    for t in range(T):

        D = np.diag(sigmas[t])
        Ht[t] = np.dot(np.dot(D, Rt[t]), D)

    return Ht


# 5. Plot dynamic correlations

def plot_dynamic_correlations(Rt, asset_names):

    for i in range(len(asset_names)):

        for j in range(i + 1, len(asset_names)):

            plt.figure(figsize = (16, 8))

            plt.plot(Rt[:, i, j])
            plt.title(f"DCC Correlation: {asset_names[i]} vs {asset_names[j]}")
            plt.grid(alpha = 0.3)
            plt.tight_layout()
            plt.savefig(eval_pred_dccgarch_path / f"dcc_corr_{asset_names[i]}_{asset_names[j]}.png")
            plt.close()

# 6. Visual output

def dcc_visual_outputs(Ht, Rt, sigmas, asset_names):

    # 1. Heatmap: Last conditional variance

    last_cov = Ht[-1]

    plt.figure(figsize = (16, 14))

    plt.imshow(last_cov, cmap = "inferno")
    plt.colorbar(label = "Covariance")
    plt.xticks(range(len(asset_names)), asset_names, rotation = 45)
    plt.yticks(range(len(asset_names)), asset_names)
    plt.title("DCC-GARCH Conditional Covariance (t+1)")
    plt.tight_layout()
    plt.savefig(out_diverse_path / "11_dccgarch_covariance_heatmap.png")
    plt.close()

    # 2. Conditional Volatilities

    plt.figure(figsize = (16, 14))

    for i, name in enumerate(asset_names):

        plt.plot(sigmas[:, i], label = name)

    plt.title("Conditional Volatilities (GARCH)")
    plt.ylabel("Volatility")
    plt.legend()
    plt.grid(alpha = 0.3)
    plt.tight_layout()
    plt.savefig(eval_pred_dccgarch_path / "dcc_conditional_volatilities.png")
    plt.close()

    # 3. Dynamic correlations (DCC)

    for i in range(len(asset_names)):

        for j in range(i + 1, len(asset_names)):

            plt.figure(figsize = (16, 14))
            plt.plot(Rt[:, i, j])
            plt.title(f"DCC Correlation: {asset_names[i]} vs {asset_names[j]}")
            plt.ylabel("Correlation")
            plt.grid(alpha = 0.3)
            plt.tight_layout()
            plt.savefig(
                eval_pred_dccgarch_path / f"dcc_corr_{asset_names[i]}_{asset_names[j]}.png"
            )
            plt.close()

    print("DCC-GARCH visual outputs saved.")



# ------------------------------------------------------------------------------------------------------------

# Main


if __name__ == "__main__":

    print("DCC-GARCH Pipeline...")

    # 1. Data preparation

    train, val, test, assets = load_and_prepare_multivariate_log_returns(top_k = 5)

    # 2. Univariate GARCH

    results, sigmas, Z = fit_univariate_garch(train, p = 1, q = 1, distribution = "student_t")

    # 3. DCC

    Rt, Qt = estimate_dcc(Z, a = 0.05, b = 0.93)

    # 4. Covariance matrix

    Ht = compute_conditional_covariance(sigmas, Rt)

    # 5. Plot dymanic correlations

    plot_dynamic_correlations(Rt, assets)

    # 6. Outputs

    dcc_visual_outputs(Ht, Rt, sigmas, assets)


    print("DCC-GARCH pipeline finished.")