# Details of  consolidation script:

# Load the 3 main dataframes (SPY_Close, Features from FRED and Technical Indicators).
# The script merge them into a single dataframe and get unified as a time series.
# Impute the missing values --> Eliminating the first 222 rows and applying the ffill() method.
# Split the dataframe in 3 subsets (train set, validation set and test set) --> Used for time series.
# Scale via Robust Scaler to standardize the data (mostly for deep learning methods).
# Save the 3 resulting dataframes in csv format into src/data/prepared
# Save the SPY close price in csv format into src/data/prepared



# Libraries and packages

import os
import pandas as pd
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from sklearn.preprocessing import RobustScaler
import joblib


# Adjust directories (sources and outputs)

load_dotenv()

raw_data_path = os.getenv("RAW_DATA_PATH")
processed_data_path = os.getenv("PROCESSED_DATA_PATH")
prepared_data_path = os.getenv("PREPARED_DATA_PATH")
out_object_path = os.getenv("OUT_OBJECTS_PATH")

raw_data_path = Path(raw_data_path)
processed_data_path = Path(processed_data_path)
prepared_data_path = Path(prepared_data_path)
out_object_path = Path(out_object_path)



# Function to load, merge the data (3 dataframes --> SPY_Close, Features and Technical indicators) and impute the missing values


def load_merge_impute():


    print("Loading datasets...")


    # Load SPY (Close Price)

    spy = pd.read_csv(raw_data_path / 'SPY_raw_data.csv', header = 0)

    spy = spy.iloc[2:].reset_index(drop = True)
    spy = spy.rename(columns = {spy.columns[0]: 'Date'})
    spy['Date'] = pd.to_datetime(spy['Date'])
    spy = spy.set_index('Date')
    spy = spy.apply(pd.to_numeric, errors = 'coerce')

    spy_close = spy['Close']

    spy_close.index = pd.to_datetime(spy_close.index)
    spy_close = pd.DataFrame(spy_close)

    # Rename Close column

    spy_close = spy_close.rename(columns = {'Close': 'Close_SPY'})

    print("SPY Close loaded")


    # Load the Features Data

    complete_features = pd.read_csv(processed_data_path / 'features_processed_data.csv')
    complete_features = complete_features.copy()

    if 'Date' in complete_features.columns:
        complete_features['Date'] = pd.to_datetime(complete_features['Date'])
        complete_features = complete_features.set_index('Date')

    print("Features loaded")

    # Load the Technical Indicators data

    complete_technicals = pd.read_csv(processed_data_path / 'technical_indicators_processed_data.csv')
    complete_technicals = complete_technicals.copy()

    if 'Date' in complete_technicals.columns:
        complete_technicals['Date'] = pd.to_datetime(complete_technicals['Date'])
        complete_technicals = complete_technicals.set_index('Date')


    print("Technical Indicators loaded")


    # Merging process

    df_final = (
        spy_close
        .join(complete_features, how = "left")
        .join(complete_technicals, how = "left")
    )

    # Order by data and make sure the index

    df_final = df_final.sort_index()
    df_final.index.name = "Date"

    print("Datasets merged into one")


    # Imputation process --> First, drop the first 222 rows and if there are more missing values, ffill() will be applied

    df_final = df_final[222:]

    df_final = df_final.ffill()


    print("Dataset imputed")


    return df_final


# Function to scale and split the data in 3 subsets (train, validation and evaluation)


def scale_split(df, target_column = 'Close_SPY'):


    # Sort the dataframe by date (checkup)

    df = df.sort_index()

    # Define cuts (80% for train, 10% for validation, 10% for evaluation)

    n = len(df)
    train_end = int(n * 0.8)
    validation_end = int(n * 0.9)

    train = df.iloc[:train_end]
    validation = df.iloc[train_end:validation_end]
    testing = df.iloc[validation_end:]

    print(f"    - Train: {train.shape[0]} trading days")
    print(f"    - Validation:   {validation.shape[0]} trading days")
    print(f"    - Testing:  {testing.shape[0]} trading days")

    print("Division Finished")


    # Scaling with Robust Scaler (handles better outliers and spikes) --> Median / IQR
    # Fit only in train subset

    features_to_scale = [c for c in df.columns if c != target_column]

    scaler_features = RobustScaler()
    scaler_features.fit(train[features_to_scale])

    # Scaler for target

    scaler_target = RobustScaler()
    scaler_target.fit(train[[target_column]])

    # Transform in all subsets

    train_scaled = train.copy()
    val_scaled = validation.copy()
    test_scaled = testing.copy()

    train_scaled[features_to_scale] = scaler_features.transform(train[features_to_scale])
    val_scaled[features_to_scale] = scaler_features.transform(validation[features_to_scale])
    test_scaled[features_to_scale] = scaler_features.transform(testing[features_to_scale])

    train_scaled[target_column] = scaler_target.transform(train[[target_column]]).flatten()
    val_scaled[target_column]   = scaler_target.transform(validation[[target_column]]).flatten()
    test_scaled[target_column]  = scaler_target.transform(testing[[target_column]]).flatten()

    print("Subsets scaled (Features & Target)")

    # Save the scalers with joblib

    scaler_features_path = os.path.join(out_object_path, 'features_robust_scaler.joblib')
    joblib.dump(scaler_features, scaler_features_path)

    scaler_target_path = os.path.join(out_object_path, 'target_robust_scaler.joblib')
    joblib.dump(scaler_target, scaler_target_path)

    print("Scalers saved")

    return train_scaled, val_scaled, test_scaled


# Function to save datasets (train, validation and test)

def save_datasets(train, validation, test):

    train.to_csv(prepared_data_path / 'train_dataset.csv')
    validation.to_csv(prepared_data_path / 'validation_dataset.csv')
    test.to_csv(prepared_data_path / 'test_dataset.csv')

    print("Saved files")




# Main 

if __name__ == '__main__':

    # Load, merge and impute

    master_df = load_merge_impute()

    # Split and scale

    train_df, validation_df, test_df = scale_split(master_df, target_column = 'Close_SPY')

    # Save datasets

    save_datasets(train_df, validation_df, test_df)

    print("Successful Consolidation")