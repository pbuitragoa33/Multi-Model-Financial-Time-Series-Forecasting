# Sript to obtain other relevant economic features from FRED database

import pandas as pd
import os
from fredapi import Fred
from dotenv import load_dotenv


# Load environment variabels

load_dotenv()

raw_data_paths = os.getenv('RAW_DATA_PATH')
FRED_API_KEY = os.getenv("FRED_API_KEY")


# Initialize the FRED client

fred = Fred(api_key = FRED_API_KEY)



# Function to get features from FRED

def get_additional_features_FRED(serie_id, start_date = '2005-01-01', end_date = '2025-11-21'):

    data = fred.get_series(series_id = serie_id, observation_start = start_date, observation_end = end_date)

    data = data.to_frame()
    data = data.reset_index()

    data.columns = ['Date', serie_id]


    return data



# Usage: List of ids of relevant economic indicators from FRED

# - ICE BofA US High Yield Index Option-Adjusted Spread (BAMLH0A0HYM2)
# - ICE BofA 7-10 Year US Corporate Bond Index Effective Yield (BAMLC4A0C710YEY)
# - National Financial Conditions Index (NFCI)  -- weekly
# - St. Louis Fed Financial Stress Index (STLFSI4)  -- weekly
# - 5 Year Breakeven Inflation Rate(T5YIE)
# - 10 Year Treasury Constant Maturity Minus 2 Year Treasury Constant Maturity (T10Y2Y)
# - 10 Year Treasury Constant Maturity Minus 3 Month Treasury Constant Maturity (T10Y3M)
# - Moody's Seasoned Baa Corporate Bond Yield Relative to Yield on 10 Year Treasuty Constant Maturity (BAA10Y)
# - Effective Federal Funds Rate (EFFR)

indicators = ['BAMLH0A0HYM2', 'BAMLC4A0C710YEY', 'NFCI', 'STLFSI4', 'T5YIE', 'T10Y2Y', 'T10Y3M', 'BAA10Y', 'EFFR']


# Get ans save the features from FRED

for serie in indicators:

    data = get_additional_features_FRED(serie)

    file_path = os.path.join(raw_data_paths, f'{serie}_additional_feature_FRED.csv')

    data.to_csv(file_path)



# ------------------------------------------------------------------------
# Names of the saved files will be renamed later for practical purposes.
# ------------------------------------------------------------------------
