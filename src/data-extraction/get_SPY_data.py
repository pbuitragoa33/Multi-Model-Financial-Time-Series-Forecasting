# Script to obtain SPY data for the SP500 Vanguard ETF


# Libraries

import yfinance as yf
import pandas as pd
import os


# Load environment variables

from dotenv import load_dotenv

load_dotenv()

raw_data_paths = os.getenv('RAW_DATA_PATH')


# Get the SPY data (SP500 State Street ETF)

ticker = 'SPY'
start = '2005-01-01'
end = '2025-11-21'
frecuency = '1d'

spy_data = yf.download(tickers = ticker, 
                       start = start, 
                       end = end, 
                       interval = frecuency, 
                       auto_adjust = True)


# Save the data to a CSV file 

file_path = os.path.join(raw_data_paths, 'SPY_raw_data.csv')

spy_data.to_csv(file_path)