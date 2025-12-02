# Target Processing - SPY Close Price

# - Load the dataset of SPY
# - Drop the first 2 rows of features loaded from yf
# - Rename column 'Date' and convert it into datetime
# - Set 'Date' as index
# - Apply the numeric transformation for datasets
# - Calculate the Log-Returns of SPY Close Price
# - Calculate a bunch of technical indicators for SPY (with the OHLCV data)


# Save the processed target data for model training:

# - The Close Price of SPY
# - The Log-Returns of SPY Close Price

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


# Main feature - S&P 500 ETF (SPY)

spy = pd.read_csv(raw_data_path / 'SPY_raw_data.csv', header = 0)

spy = spy.iloc[2:].reset_index(drop = True)
spy = spy.rename(columns = {spy.columns[0]: 'Date'})
spy['Date'] = pd.to_datetime(spy['Date'])
spy = spy.set_index('Date')
spy = spy.apply(pd.to_numeric, errors = 'coerce')




