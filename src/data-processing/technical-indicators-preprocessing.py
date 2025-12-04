# Derived Technical Indicators from OHLCV data


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


# SPY Load

# Main feature - S&P 500 ETF (SPY)

spy = pd.read_csv(raw_data_path / 'SPY_raw_data.csv', header = 0)

spy = spy.iloc[2:].reset_index(drop = True)
spy = spy.rename(columns = {spy.columns[0]: 'Date'})
spy['Date'] = pd.to_datetime(spy['Date'])
spy = spy.set_index('Date')
spy = spy.apply(pd.to_numeric, errors = 'coerce')

print(spy.info())
print("--" * 30)
print(spy.isnull().sum())
print("--" * 30)

spy.head()


# --------------------------------- Derived Technical Indicators ------------------------------


# 1. Scaled Simple Moving Average (Scaled SMA) 

# Scaled Simple Moving Average (Close - SMA)

def scaled_SMA(df, period):

    sma = df['Close'].rolling(period).mean()
    scaled_sma = df['Close'] - sma

    return scaled_sma

# 2. Scaled Exponential Moving Average (Scaled EMA) 

# Scaled Exponential Moving Average (Close - EMA)

def scaled_EMA(df, period):

    ema = df['Close'].ewm(span = period, adjust = False).mean()
    scaled_ema = df['Close'] - ema

    return scaled_ema

# 3. Scaled Hull Moving Average (Scaled HMA) 

# Scaled Hull Moving Average (Close - HMA)

# First must be calculated the WMA, but inside the HMA function

def scaled_HMA(df, period):

    # WMA

    def WMA_component(series, length):

        weights = np.arange(1, length + 1)
        result = series.rolling(window = length)
        result = result.apply(lambda x: np.dot(x, weights) / weights.sum(), raw = True)

        return result
    
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))

    wma1 = WMA_component(df['Close'], half)
    wma2 = WMA_component(df['Close'], period)

    hma = WMA_component(2 * wma1 - wma2, sqrt_period)
    scaled_hma = df['Close'] - hma

    return scaled_hma

# 4. Momentum

# Momentum Indicator

def momentum(df, period):

    momtm = df['Close'] - df['Close'].shift(period)

    return momtm

# 5. Relative Strength Index (RSI)

# RSI (Relative Strength Index)

def rsi(df, period):

    delta = df['Close'].diff()
    
    gain = delta.clip(lower = 0)
    loss = - delta.clip(upper = 0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss

    rsi_value = 100 - (100 / (1 + rs))

    return rsi_value

# 6. Stochastic Oscillator (%K and %D)

# Stochastic Oscillator (%K and %D)

def stochastic(df, period, smooth_k = 1, smooth_d = 3):

    low_min = df['Low'].rolling(period).min()
    high_max = df['High'].rolling(period).max()

    k = 100 * (df['Close'] - low_min) / (high_max - low_min)

    k_smooth = k.rolling(smooth_k).mean()
    d_smooth = k_smooth.rolling(smooth_d).mean()

    return k_smooth, d_smooth

# 7. Williams %R

# Williams %R

def williams_r(df, period):

    low_min = df['Low'].rolling(period).min()
    high_max = df['High'].rolling(period).max()

    wr = - 100 * (high_max - df['Close']) / (high_max - low_min)

    return wr

# 8. Normalized Average True Range (NATR)

# Normalized ATR (ATR / Close)

def normalized_atr(df, period):

    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())

    tr = pd.concat([high_low, high_close, low_close], axis = 1).max(axis = 1)
    atr = tr.rolling(period).mean()

    norm_atr = atr / df['Close']

    return norm_atr

# 9. Scaled Bollinger Bands

# Scaled Bollinger Bands (with 2 standard deviations)

def scaled_bb(df, period, num_std = 2):

    sma = df['Close'].rolling(period).mean()
    std = df['Close'].rolling(period).std()

    upper = sma + (num_std * std)
    lower = sma - (num_std * std)

    scaled_upper = df['Close'] - upper
    scaled_lower = df['Close'] - lower
    
    return scaled_upper, scaled_lower

# 10. Scaled Keltner Channels

# Scaled Keltner Channels

def scaled_keltner(df, period, atr_mult = 2):

    ema = df['Close'].ewm(span = period, adjust = False).mean()

    # ATR

    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())

    tr = pd.concat([high_low, high_close, low_close], axis = 1).max(axis = 1)
    atr = tr.rolling(period).mean()

    upper = ema + (atr_mult * atr)
    lower = ema - (atr_mult * atr)

    scaled_upper = df['Close'] - upper
    scaled_lower = df['Close'] - lower

    return scaled_upper, scaled_lower

# 11. On-Balance Volume (OBV)

# On-Balance Volume

def obv(df):

    direction = np.sign(df['Close'].diff()).fillna(0)

    dir_vol = (direction * df['Volume']).cumsum()

    return dir_vol

# 12. Anchored Volume Weighted Average Price (Anchored VWAP)

# Anchored VWAP 

def anchored_vwap(df, anchor_index = 0):

    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    cum_tp_vol = (typical_price * df['Volume']).cumsum() - (typical_price * df['Volume']).cumsum().iloc[anchor_index]
    cum_vol = df['Volume'].cumsum() - df['Volume'].cumsum().iloc[anchor_index]

    vwap = cum_tp_vol / cum_vol
    
    return vwap

# 13. Intraday Logarithmic Volatility

# Intraday Logarithmic Volatility

def ilv(df):

    dlog = np.log(df['High'] / df['Low'])

    return dlog



# --------------------------------- Applying Technical Indicators ------------------------------


# Add the indicators to SPY dataframe

def add_indicators(spy, 
                   period_sma = 50,
                   period_sma2 = 200,
                   period_ema = 50,
                   period_ema2 = 200,
                   period_hma = 50,
                   period_hma2 = 200,
                   period_momentum = 20,
                   period_momentum2 = 100,
                   period_rsi = 14,
                   period_stochastic = 14,
                   period_williamsR = 21,
                   period_atr = 14,
                   period_bb = 21,
                   period_keltner = 21,
                   ):
    

    # SSMA50 and SSMA200

    spy['Scaled_SMA50'] = scaled_SMA(spy, period = period_sma)
    spy['Scaled_SMA200'] = scaled_SMA(spy, period = period_sma2)

    # SEMA50 and SEMA200

    spy['Scaled_EMA50'] = scaled_EMA(spy, period = period_ema)
    spy['Scaled_EMA200'] = scaled_EMA(spy, period = period_ema2)

    # SHMA50 and SHMA200

    spy['Scaled_HMA50'] = scaled_HMA(spy, period = period_hma)
    spy['Scaled_HMA200'] = scaled_HMA(spy, period = period_hma2)

    # Momentum

    spy['Momentum_20p'] = momentum(spy, period = period_momentum)
    spy['Momentum_100p'] = momentum(spy, period = period_momentum2)

    # RSI

    spy['RSI'] = rsi(spy, period = period_rsi)

    # Stochastic (%K and %D)

    k, d = stochastic(spy, period = period_stochastic)
    spy['Stoch_K'] = k
    spy['Stoch_D'] = d

    # Williams %R

    spy['WilliamsR'] = williams_r(spy, period = period_williamsR)

    # NATR

    spy['Norm_ATR'] = normalized_atr(spy, period = period_atr)

    # Scaled Bollinger BAnds

    s_upper, s_lower = scaled_bb(spy, period_bb)
    spy['Scaled_Upper_Bollinger'] = s_upper
    spy['Scaled_Lower_Bollinger'] = s_lower

    # Scaled Keltner Channels

    s_upper, s_lower = scaled_keltner(spy, period_keltner)
    spy['Scaled_Upper_Keltner'] = s_upper
    spy['Scaled_Lower_Keltner'] = s_lower

    # OBV

    spy['OBV'] = obv(spy)

    # Anchored VWAP

    spy['Anchored_VWAP'] = anchored_vwap(spy)

    # Intraddy Logarithmic Volatility

    spy['ILV'] = ilv(spy)


    return spy



# Add indicators --> Invoke the fuction

spy = add_indicators(spy)


# The processing of the target (Close Price of SPY) was done in the target-processing.py file
# This file contains other features

# Drop Close

spy_features = spy.drop(columns = 'Close', axis = 1)
spy_features.tail(10)


# Save the processed features data into a csv file at processed_path_data


data_features_path = os.path.join(processed_data_path, 'technical_indicators_processed_data.csv')
spy_features.to_csv(data_features_path)