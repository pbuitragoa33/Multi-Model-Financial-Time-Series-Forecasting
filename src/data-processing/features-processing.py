# Features Processing Complete Script

# Testing the following process: Loading, Merging, Wrangling and Consolidating Data

# - Load the datasets (features from yf and FRED)
# - Drop the first 2 rows of features loaded from yf
# - Rename column 'Date' and convert it into datetime
# - Set 'Date' as index
# - Apply the numeric transformation for datasets
# - Drop the unnecessary columns for FRED datasets 
# - Keep only the 'Close' price columns for assets (not SPY data) 
# - Merge the features dataframe (data_fin and data_FRED)


# Necessary libraries

import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from pathlib import Path


# Adjust directories (sources and outputs)

load_dotenv()

raw_data_path = os.getenv("RAW_DATA_PATH")
processed_data_path = os.getenv("PROCESSED_DATA_PATH")

raw_data_path = Path(raw_data_path)
processed_data_path = Path(processed_data_path)



# --------------------------- Yahoo Finance Features ----------------------------


# Function to load and clean CSV files of each ticker

def load_and_clean_tickers(path):

    df = pd.read_csv(path, header = 0)

    df = df.iloc[2:].reset_index(drop=True)
    df = df.rename(columns={df.columns[0]: 'Date'})
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date')
    df = df.apply(pd.to_numeric, errors = 'coerce')

    df = df['Close'].to_frame()

    return df 


# Invoke the fuction

tickers = {
    "vix": "VIX_raw_data.csv",
    "gold": "Gold_raw_data.csv",
    "oil": "CrudeOil_raw_data.csv",
    "tlt": "TLT_raw_data.csv",
    "rsp": "RSP_raw_data.csv",
    "tnx": "TNX_raw_data.csv",
    "iwm": "IWM_raw_data.csv",
    "dxy": "DXY_raw_data.csv"
}

data_fin = {name: load_and_clean_tickers(raw_data_path / file) for name, file in tickers.items()}


# Convert the dictionary into a dataframe

data_fin = pd.concat(
    {k: v.rename(columns = {'Close': k}) for k, v in data_fin.items()},
    axis = 1
)

data_fin.columns = data_fin.columns.droplevel(0)
data_fin

data_fin.info()



# --------------------------- FRED (Federal Reserve) Features ----------------------------


# Function to load and clean CSV files from FRED sources

def load_csv_FRED(path, column):

    df = pd.read_csv(path, parse_dates=['Date'], index_col = 'Date')

    return df[[column]]


# Invoke the function 

datasets_FRED = {
    "baa10yc": ("Baa_Corporate_to_10_Yield.csv", "BAA10Y"),
    "corp710y": ("Corporate_Bond_710_raw_data.csv", "BAMLC4A0C710YEY"),
    "nfci": ("NFCI_fin_condition_raw_data.csv", "NFCI"),
    "str_index": ("STLFSI4_Stress_raw_data.csv", "STLFSI4"),
    "t5yie": ("T5YIE_Breakeven_raw_data.csv", "T5YIE"),
    "t10y2y": ("T10Y_minus_2Y_raw_data.csv", "T10Y2Y"),
    "t10y3m": ("T10Y_minus_3M_raw_data.csv", "T10Y3M"),
    "effr": ("EFFR_funds_rates_raw_data.csv", "EFFR"),
    "high_yield": ("High_Yield_raw_data.csv", "BAMLH0A0HYM2")
}

data_FRED = {name: load_csv_FRED(raw_data_path / file[0], file[1]) for name, file in datasets_FRED.items()}


# Convert the dictionary into a dataframe

data_FRED = pd.concat(
    {k: v.rename(columns = {'Close': k}) for k, v in data_FRED.items()},
    axis = 1
)

data_FRED.columns = data_FRED.columns.droplevel(0)
data_FRED

data_FRED.info()



# --------------------------- Merging Features DataFrames ----------------------------


"""
The "NFCI (National Financial Condition Index)" and "STLFSI4 (Stress Level Index)" columns contain a large number of null (NaN) values. This occurs because both series are originally reported at a weekly frequency, so when they are incorporated into a dataset with daily frequency, gaps naturally appear on the days between observations.

To properly handle these missing values, the following procedure will be applied:

  * Apply forward filling (ffill()) to propagate the most recent available value forward until a new weekly observation is encountered.

This approach preserves the temporal structure of the original weekly data while adapting the series to the daily frequency required for the analysis, without introducing artificial or inconsistent values. With that, the data will talk the same language, daily frequency, but not exactly because the market stays close in weekends, holidays and other celebrations.
"""

# Apply the previous strategy to manage the columns with different frequencies (using forward fill)

def apply_ffill(df, weekly_columns):

    # Make sure the index is in datetime format
    
    df = df.copy()
    df.index = pd.to_datetime(df.index)

    # Aplly the forward fill

    df[weekly_columns] = df[weekly_columns].ffill()

    return df


# Invoke the function

# For data_fin

# There are no weekly columns in data_fin

# For data_FRED

weekly_cols_FRED = ['NFCI', 'STLFSI4']   

data_FRED = apply_ffill(df = data_FRED, weekly_columns = weekly_cols_FRED)
data_FRED


# The next step is to consolidate the features dataframe, which will contain
# data_fin and data_FRED dataframes, without cleaning and processign the final version of features.

# Merging all the data (data_fin and data_FRED) and consolidating into a single dataframe (data_features)

data_features = data_fin.join(data_FRED, how = 'left')

print(data_features.isnull().sum())
print("--" * 30)
data_features


# Save the processed features data into a csv file at processed_path_data

data_features_path = os.path.join(processed_data_path, 'features_processed_data.csv')
data_features.to_csv(data_features_path)

