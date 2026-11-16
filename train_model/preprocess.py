"""Load the churn datasets and turn raw tables into model-ready features.

Preprocessing has two learned parts: the median used to fill each numeric
column's missing values, and an OrdinalEncoder for the text columns. Both
are fitted on the training split only and saved together in encoder.pkl,
so the test split and live requests are transformed with exactly the same
values and nothing from the test data leaks into training.
"""

import os
import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

from settings import (  # isort:skip
    DATA_DIR,
    DATASET_NUM,
    DEBUG,
    RANDOM_STATE,
    REMOVE_OUTLIERS,
    TARGET,
    TEST_SIZE,
)

# Identifier and timestamp columns that carry no predictive signal.
USELESS_COLUMNS = ['CustomerID', 'customerID', 'UpdatedAt']

# Columns whose outliers can optionally be removed (spelling differs per dataset).
OUTLIER_COLUMNS = ['Tenure', 'tenure']


def model_dir(dataset_num=DATASET_NUM):
    """Return the folder that holds the model and encoder for a dataset."""
    return f'./model/{dataset_num}/'


def load_dataset(dataset_num=DATASET_NUM):
    """Read the raw churn table for dataset 1 (e-commerce) or 2 (telco).

    Returns an empty DataFrame if the dataset number is unknown.
    """
    print(f'\nLoading dataset [{dataset_num}]...')
    if dataset_num == 1:
        data = pd.read_excel(DATA_DIR + 'E Commerce Dataset.xlsx', sheet_name='E Comm')
    elif dataset_num == 2:
        data = pd.read_csv(DATA_DIR + 'dqlab_telco_final.csv')
    else:
        print('Load data error: unknown dataset number:', dataset_num)
        return pd.DataFrame()

    print(f' Loaded {data.shape[0]} records.')
    if DEBUG:
        data.info()
        print(data.head())
    return data


def determine_outlier_thresholds_iqr(df, col_name, th1=0.25, th3=0.75):
    """Return (lower, upper) limits for a column using the interquartile range.

    Values outside quartile1 - 1.5*IQR and quartile3 + 1.5*IQR count as outliers.
    """
    quartile1 = df[col_name].quantile(th1)
    quartile3 = df[col_name].quantile(th3)
    iqr = quartile3 - quartile1
    return quartile1 - 1.5 * iqr, quartile3 + 1.5 * iqr


def determine_outlier_thresholds_sdm(df, col_name, scale):
    """Return (lower, upper) limits as mean -/+ `scale` standard deviations."""
    mean = df[col_name].mean()
    std = df[col_name].std()
    return mean - scale * std, mean + scale * std


def print_missing_values_table(data):
    """Print the count and percentage of missing values per column."""
    na_columns = [col for col in data.columns if data[col].isnull().sum() > 0]
    n_miss = data[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (n_miss / data.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df)


def preprocess_df(data, fill_values=None):
    """Clean a raw table: drop ID columns, encode the label and fill missing values.

    Args:
        data: raw DataFrame, with or without the TARGET column.
        fill_values: per-column medians to fill missing numeric values with.
            Pass None when cleaning training data, so the medians are computed
            from `data`; pass the training medians for test or live data.

    Returns:
        (cleaned DataFrame, the fill values that were used).
    """
    print('\nPreprocessing data...')
    total_rows = data.shape[0]
    data = data.drop(columns=[col for col in USELESS_COLUMNS if col in data.columns])

    # Dataset 2 stores the label as 'Yes'/'No'; convert it to 1/0 before encoding.
    if TARGET in data.columns and data[TARGET].dtype == object:
        data[TARGET] = data[TARGET].map({'No': 0, 'Yes': 1}).astype(int)

    if DEBUG and TARGET in data.columns and len(data) > 10:
        categorical = data.select_dtypes(include='object').columns
        print(f'\nCategorical columns: {list(categorical)}')
        corr = data.corr(numeric_only=True)[TARGET]
        print(f'\nCorrelation to {TARGET}:\n{corr.to_string()}')
        print_missing_values_table(data)

    # Fill missing numeric values with medians (learned on training data only).
    if fill_values is None:
        fill_values = data.drop(columns=[TARGET], errors='ignore').median(numeric_only=True)
    # infer_objects turns columns that held None (e.g. JSON null) back into numbers.
    data = data.fillna(fill_values).infer_objects()

    if REMOVE_OUTLIERS:
        for col in OUTLIER_COLUMNS:
            if col in data.columns:
                lower, upper = determine_outlier_thresholds_iqr(data, col, th1=0.05, th3=0.95)
                print(f'\nRemoving {col} outliers outside [{lower}, {upper}] (IQR method)')
                data = data[(data[col] >= lower) & (data[col] <= upper)]

    print(f'Kept {data.shape[0]} / {total_rows} records.')
    return data, fill_values


def enc_save(enc, file_name):
    """Save a fitted encoder (with its fill values) to a pickle file."""
    with open(file_name, 'wb') as f:
        pickle.dump(enc, f)


def enc_load(file_name):
    """Load an encoder saved by enc_save."""
    with open(file_name, 'rb') as f:
        return pickle.load(f)


def preprocess_data(df, ord_enc, dataset_num=DATASET_NUM, fit_enc=False):
    """Turn a raw table into numeric features ready for the model.

    Args:
        df: raw DataFrame.
        ord_enc: an OrdinalEncoder. With fit_enc=True it is fitted on `df`
            (training data) and saved; otherwise its saved fitting is reused.
        dataset_num: which dataset `df` comes from (decides where to save).
        fit_enc: True for training data, False for test data and requests.

    Returns:
        The preprocessed DataFrame, with every column numeric.
    """
    # Reuse the training medians if the encoder carries them; older saved
    # encoders do not, and then the medians of `df` itself are used.
    fill_values = None if fit_enc else getattr(ord_enc, 'fill_values_', None)
    df, fill_values = preprocess_df(df, fill_values)

    if fit_enc:
        categorical_features = list(df.select_dtypes(exclude=[np.number]).columns)
    else:
        # Use the encoder's own column list and order, so input field order does not matter.
        categorical_features = list(getattr(ord_enc, 'feature_names_in_', []))
    if categorical_features:
        if fit_enc:
            df[categorical_features] = ord_enc.fit_transform(df[categorical_features])
            ord_enc.fill_values_ = fill_values  # stored with the encoder for reuse
            os.makedirs(model_dir(dataset_num), exist_ok=True)
            enc_save(ord_enc, f'{model_dir(dataset_num)}encoder.pkl')
            if DEBUG:
                print(' OrdinalEncoder categories:', ord_enc.categories_)
        else:
            df[categorical_features] = ord_enc.transform(df[categorical_features])
    return df


def new_encoder():
    """Create an OrdinalEncoder that maps categories unseen in training to -1."""
    return OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)


def load_split_data(dataset_num=DATASET_NUM, test_size=TEST_SIZE, random_state=RANDOM_STATE):
    """Load a dataset, split it, and preprocess both parts without leakage.

    The split happens on the raw data. The fill values and the encoder are
    fitted on the training part only and then applied to the test part.

    Returns:
        (train_df, test_df), both preprocessed.
    """
    raw = load_dataset(dataset_num)
    if raw.empty:
        return pd.DataFrame(), pd.DataFrame()

    train_raw, test_raw = train_test_split(
        raw, test_size=test_size, random_state=random_state, stratify=raw[TARGET]
    )
    ord_enc = new_encoder()
    train_df = preprocess_data(train_raw.copy(), ord_enc, dataset_num, fit_enc=True)
    test_df = preprocess_data(test_raw.copy(), ord_enc, dataset_num, fit_enc=False)
    return train_df, test_df


def load_data(dataset_num=DATASET_NUM):
    """Load and preprocess a full dataset, fitting the encoder on all of it.

    Use this only for exploration and the drift demo. For training and
    evaluation use load_split_data, which keeps the test rows unseen.
    """
    df = load_dataset(dataset_num)
    if df.empty:
        return df
    return preprocess_data(df, new_encoder(), dataset_num, fit_enc=True)


if __name__ == '__main__':
    # Quick manual check: preprocess both datasets and show the result.
    for num in [1, 2]:
        train, test = load_split_data(num)
        print(f'\nDataset {num}: train {train.shape}, test {test.shape}')
        print(train.head(3).to_string())
