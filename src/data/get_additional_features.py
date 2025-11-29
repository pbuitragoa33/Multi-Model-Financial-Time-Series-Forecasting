# Script to obtain additional features for financial data


# Libraries

import yfinance as yf
import pandas as pd
import os


# Load environment variables

from dotenv import load_dotenv

load_dotenv()

raw_data_paths = os.getenv('RAW_DATA_PATH')


# Function to get additional features for some tickers

def get_additional_features(ticker, 
                            start = '2005-01-01', 
                            end = '2025-11-21', 
                            frequency = '1d'):

    data = yf.download(tickers = ticker, 
                       start = start, 
                       end = end, 
                       interval = frequency, 
                       auto_adjust = True)

    return data



# Usage. For this project we will get the following additional features:

# - Volatility Index (^VIX)
# - Gold Futures (GC=F)
# - Crude Oil Futures (CL=F)
# - iShares 20+ Year Treasury Bond ETF (TLT)
# - Invesco S&P 500 Equal Weight ETF (RSP)
# - 10-Year Treasury Note (^TNX)
# - iShares Russell 2000 ETF (IWM)
# - US Dollar Index (DX-Y-NYB)

additional_tickers = ['^VIX', 'GC=F', 'CL=F', 'TLT', 'RSP', 'IWM', 'DX-Y.NYB']


# Get and save the additional features except TNX 

for ticker in additional_tickers:

    data = get_additional_features(ticker)

    file_path = os.path.join(raw_data_paths, f'{ticker}_additional_feature.csv')

    data.to_csv(file_path)


# TNX (10-Year Trasury Note) need to be treated separately due to its initial character '^'

tnx_data = get_additional_features('^TNX')

tnx_file_path = os.path.join(raw_data_paths, 'TNX_additional_feature.csv')
tnx_data.to_csv(tnx_file_path)



# ------------------------------------------------------------------------
# Names of the saved files will be renamed later for practical purposes.
# ------------------------------------------------------------------------