# Load every CSV file in the 'data/raw' directory to plot them together and save the plots
# as PNG files in the 'outputs/initial-visualizations' directory.


# Libraries

import os
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from dotenv import load_dotenv


# Load environment variables

load_dotenv()

INIT_VIS_PATH = Path(os.getenv('INITIAL_VISUALIZATIONS_PATH'))
RAW_DATA_PATH = Path(os.getenv('RAW_DATA_PATH'))

# Ensure output directory exists

INIT_VIS_PATH.mkdir(parents = True, exist_ok = True)


# Extract all CSV files from the raw data directory

raw_data_dir = Path(RAW_DATA_PATH)

csv_files = list(raw_data_dir.glob('*.csv'))

print("Found ", len(csv_files), "CSV files in the directory.")


# Plot each CSV file and save the plots as PNG files


# 1. Moody's Seasoned Baa Corporate Bond Yield Relative to Yield on 10 Year Treasuty Constant Maturity 

baa10yc = pd.read_csv(raw_data_dir / 'Baa_Corporate_to_10_Yield.csv', parse_dates = ['Date'], index_col = 'Date')

plt.figure(figsize = (16, 12))
plt.plot(baa10yc.index, baa10yc['BAA10Y'], label = "Moody's Baa Corporate Bond Yield Spread", color = 'blue')
plt.title(" Moody's Seasoned Baa Corporate Bond Yield Relative to Yield on 10 Year Treasuty Constant Maturity")
plt.xlabel("Date")
plt.ylabel("Yield Spread in %")
plt.legend()
plt.grid()
plt.savefig(INIT_VIS_PATH / 'Baa_Corporate_to_10_yield.png')
plt.close()

# 2. ICE BofA 7-10 Year US Corporate Bond Index Effective Yield

corp710y = pd.read_csv(raw_data_dir / 'Corporate_Bond_710_raw_data.csv', parse_dates = ['Date'], index_col = 'Date')

plt.figure(figsize = (16, 12))
plt.plot(corp710y.index, corp710y['BAMLC4A0C710YEY'], label = 'ICE BofA 7-10 Year US Corporate Bond Index Effective Yield', color = 'teal')
plt.title('ICE BofA 7-10 Year US Corporate Bond Index Effective Yield')
plt.xlabel("Date")
plt.ylabel("Yield in %")
plt.legend()
plt.grid()
plt.savefig(INIT_VIS_PATH / '7-10_US_Corporate_Bond_Yield.png')
plt.close()

# 3. National Financial Conditions Index 

nfci = pd.read_csv(raw_data_dir / 'NFCI_fin_condition_raw_data.csv', parse_dates = ['Date'], index_col = 'Date')

plt.figure(figsize = (16, 12))
plt.plot(nfci.index, nfci['NFCI'], label = 'National Financial Conditions Index', color = 'olive')
plt.title('National Financial Conditions Index')
plt.xlabel("Date")
plt.ylabel("NFCI value")
plt.legend()
plt.grid()
plt.savefig(INIT_VIS_PATH / 'NFCI.png')
plt.close()

# 4. St. Louis Fed Financial Stress Index (STLFSI4)

str_index = pd.read_csv(raw_data_dir / 'STLFSI4_Stress_raw_data.csv', parse_dates = ['Date'], index_col = 'Date')

plt.figure(figsize = (16, 12))
plt.plot(str_index.index, str_index['STLFSI4'], label = 'St. Louis Fed Financial Stress Index', color = 'orangered')
plt.title('St. Louis Fed Financial Stress Index')
plt.xlabel("Date")
plt.ylabel("Stress Index value")
plt.legend()
plt.grid()
plt.savefig(INIT_VIS_PATH / 'STLFSI4_stress.png')
plt.close()


# 5. 5 Year Breakeven Inflation Rate(T5YIE)

t5yie = pd.read_csv(raw_data_dir / 'T5YIE_Breakeven_raw_data.csv', parse_dates = ['Date'], index_col = 'Date')

plt.figure(figsize = (16, 12))
plt.plot(t5yie.index, t5yie['T5YIE'], label = '5 Year Breakeven Inflation Rate', color = 'limegreen')
plt.title('5 Year Breakeven Inflation Rate')
plt.xlabel("Date")
plt.ylabel("Inflation Rate in %")
plt.legend()
plt.grid()
plt.savefig(INIT_VIS_PATH / 'T5YIE.png')
plt.close()


# 6. 10 Year Treasury Constant Maturity Minus 2 Year Treasury Constant Maturity (T10Y2Y)

t10y2y = pd.read_csv(raw_data_dir / 'T10Y_minus_2Y_raw_data.csv', parse_dates = ['Date'], index_col = 'Date')

plt.figure(figsize = (16, 12))
plt.plot(t10y2y.index, t10y2y['T10Y2Y'], label = '10 Year Treasury Constant Maturity Minus 2 Year Treasury Constant Maturity', color = 'royalblue')
plt.title('10 Year Treasury Constant Maturity Minus 2 Year Treasury Constant Maturity')
plt.xlabel("Date")
plt.ylabel("Yield Difference in %")
plt.legend()
plt.grid()
plt.savefig(INIT_VIS_PATH / 'T10Y2Y.png')
plt.close()

# 7. 10 Year Treasury Constant Maturity Minus 3 Month Treasury Constant Maturity (T10Y3M)

t10y3m = pd.read_csv(raw_data_dir / 'T10Y_minus_3M_raw_data.csv', parse_dates = ['Date'], index_col = 'Date')

plt.figure(figsize = (16, 12))
plt.plot(t10y3m.index, t10y3m['T10Y3M'], label = '10 Year Treasury Constant Maturity Minus 3 Month Treasury Constant Maturity', color = 'blueviolet')
plt.title('10 Year Treasury Constant Maturity Minus 3 Month Treasury Constant Maturity')
plt.xlabel("Date")
plt.ylabel("Yield Difference in %")
plt.legend()
plt.grid()
plt.savefig(INIT_VIS_PATH / 'T10Y3M.png')
plt.close()

# 8. Effective Federal Funds Rate (EFFR)

effr = pd.read_csv(raw_data_dir / 'EFFR_funds_rates_raw_data.csv', parse_dates = ['Date'], index_col = 'Date')

plt.figure(figsize = (16, 12))
plt.plot(effr.index, effr['EFFR'], label = 'Effective Federal Funds Rate', color = 'mediumspringgreen')
plt.title('Effective Federal Funds Rate')
plt.xlabel("Date")
plt.ylabel("Yield in %")
plt.legend()
plt.grid()
plt.savefig(INIT_VIS_PATH / 'EFFR.png')
plt.close()

# 9. ICE BofA US High Yield Index Option-Adjusted Spread (BAMLH0A0HYM2)

high_yield = pd.read_csv(raw_data_dir / 'High_Yield_raw_data.csv', parse_dates = ['Date'], index_col = 'Date')

plt.figure(figsize = (16, 12))
plt.plot(high_yield.index, high_yield['BAMLH0A0HYM2'], label = 'ICE BofA US High Yield Index Option-Adjusted Spread', color = 'deeppink')
plt.title('ICE BofA US High Yield Index Option-Adjusted Spread')
plt.xlabel("Date")
plt.ylabel("Yield in %")
plt.legend()
plt.grid()
plt.savefig(INIT_VIS_PATH / 'High_Yield.png')
plt.close()

# 10. Volatility Index (^VIX)

vix = pd.read_csv(raw_data_dir / 'VIX_raw_data.csv', header=0)

vix = vix.iloc[2:].reset_index(drop = True)
vix = vix.rename(columns = {vix.columns[0]: 'Date'})
vix['Date'] = pd.to_datetime(vix['Date'])
vix = vix.set_index('Date')
vix = vix.apply(pd.to_numeric, errors = 'coerce')

plt.figure(figsize = (16, 12))
plt.plot(vix.index, vix['Close'], label = 'Volatility index', color = 'darkred')
plt.title('VIX - Volatility Index')
plt.xlabel("Date")
plt.ylabel("VIX value")
plt.legend()
plt.grid()
plt.savefig(INIT_VIS_PATH / 'VIX.png')
plt.close()

# 11. Gold Futures

gold = pd.read_csv(raw_data_dir / 'Gold_raw_data.csv', header = 0)

gold = gold.iloc[2:].reset_index(drop = True)
gold = gold.rename(columns = {gold.columns[0]: 'Date'})
gold['Date'] = pd.to_datetime(gold['Date'])
gold = gold.set_index('Date')
gold = gold.apply(pd.to_numeric, errors = 'coerce')

plt.figure(figsize = (16, 12))
plt.plot(gold.index, gold['Close'], label = 'Gold price', color = 'gold')
plt.title('Gold Futures')
plt.xlabel("Date")
plt.ylabel("Gold price in USD")
plt.legend()
plt.grid()
plt.savefig(INIT_VIS_PATH / 'Gold.png')
plt.close()

# 12. Crude Oil Futures

oil = pd.read_csv(raw_data_dir / 'CrudeOil_raw_data.csv', header = 0)

oil = oil.iloc[2:].reset_index(drop = True)
oil = oil.rename(columns = {oil.columns[0]: 'Date'})
oil['Date'] = pd.to_datetime(oil['Date'])
oil = oil.set_index('Date')
oil = oil.apply(pd.to_numeric, errors = 'coerce')

plt.figure(figsize = (16, 12))
plt.plot(oil.index, oil['Close'], label = 'Oil price', color = 'black')
plt.title('Oil Futures')
plt.xlabel("Date")
plt.ylabel("Oil price in USD")
plt.legend()
plt.grid()
plt.savefig(INIT_VIS_PATH / 'Oil.png')
plt.close()

# 13. iShares 20+ Year Treasury Bond ETF (TLT)

tlt = pd.read_csv(raw_data_dir / 'TLT_raw_data.csv', header = 0)

tlt = tlt.iloc[2:].reset_index(drop = True)
tlt = tlt.rename(columns = {tlt.columns[0]: 'Date'})
tlt['Date'] = pd.to_datetime(tlt['Date'])
tlt = tlt.set_index('Date')
tlt = tlt.apply(pd.to_numeric, errors = 'coerce')

plt.figure(figsize = (16, 12))
plt.plot(tlt.index, tlt['Close'], label = 'TLT ETF', color = 'moccasin')
plt.title('iShares 20+ Year Treasury Bond ETF (TLT)')
plt.xlabel("Date")
plt.ylabel("TLT price in USD")
plt.legend()
plt.grid()
plt.savefig(INIT_VIS_PATH / 'TLT.png')
plt.close()

# 14. Invesco S&P 500 Equal Weight ETF (RSP)

rsp = pd.read_csv(raw_data_dir / 'RSP_raw_data.csv', header = 0)

rsp = rsp.iloc[2:].reset_index(drop = True)
rsp = rsp.rename(columns = {rsp.columns[0]: 'Date'})
rsp['Date'] = pd.to_datetime(rsp['Date'])
rsp = rsp.set_index('Date')
rsp = rsp.apply(pd.to_numeric, errors = 'coerce')

plt.figure(figsize = (16, 12))
plt.plot(rsp.index, rsp['Close'], label = 'RSP ETF', color = 'aquamarine')
plt.title('Invesco S&P 500 Equal Weight ETF')
plt.xlabel("Date")
plt.ylabel("RSP price in USD")
plt.legend()
plt.grid()
plt.savefig(INIT_VIS_PATH / 'RSP.png')
plt.close()

# 15. 10-Year Treasury Note (^TNX)

tnx = pd.read_csv(raw_data_dir / 'TNX_raw_data.csv', header = 0)

tnx = tnx.iloc[2:].reset_index(drop = True)
tnx = tnx.rename(columns = {tnx.columns[0]: 'Date'})
tnx['Date'] = pd.to_datetime(tnx['Date'])
tnx = tnx.set_index('Date')
tnx = tnx.apply(pd.to_numeric, errors = 'coerce')

plt.figure(figsize = (16, 12))
plt.plot(tnx.index, tnx['Close'], label = 'TNX ETF', color = 'plum')
plt.title('10-Year Treasury Note')
plt.xlabel("Date")
plt.ylabel("TNX in %")
plt.legend()
plt.grid()
plt.savefig(INIT_VIS_PATH / 'TNX.png')
plt.close()

# 16. iShares Russell 2000 ETF (IWM)

iwm = pd.read_csv(raw_data_dir / 'IWM_raw_data.csv', header = 0)

iwm = iwm.iloc[2:].reset_index(drop = True)
iwm = iwm.rename(columns = {iwm.columns[0]: 'Date'})
iwm['Date'] = pd.to_datetime(iwm['Date'])
iwm = iwm.set_index('Date')
iwm = iwm.apply(pd.to_numeric, errors = 'coerce')

plt.figure(figsize = (16, 12))
plt.plot(iwm.index, iwm['Close'], label = 'IWM ETF', color = 'chartreuse')
plt.title('iShares Russell 2000 ETF')
plt.xlabel("Date")
plt.ylabel("IWM price in USD")
plt.legend()
plt.grid()
plt.savefig(INIT_VIS_PATH / 'IWM.png')
plt.close()

# 17. US Dollar Index (DX-Y-NYB)

dxy = pd.read_csv(raw_data_dir / 'DXY_raw_data.csv', header = 0)

dxy = dxy.iloc[2:].reset_index(drop = True)
dxy = dxy.rename(columns = {dxy.columns[0]: 'Date'})
dxy['Date'] = pd.to_datetime(dxy['Date'])
dxy = dxy.set_index('Date')
dxy = dxy.apply(pd.to_numeric, errors = 'coerce')

plt.figure(figsize = (16, 12))
plt.plot(dxy.index, dxy['Close'], label = 'DXY Index', color = 'seagreen')
plt.title('US Dollar Index')
plt.xlabel("Date")
plt.ylabel("DXY value")
plt.legend()
plt.grid()
plt.savefig(INIT_VIS_PATH / 'DXY.png')
plt.close()


# It would be much easier to automate this process using loops, but the source files have 
# different structutres and finally I decided to write a plot and copy the next one and change
# the variable name.