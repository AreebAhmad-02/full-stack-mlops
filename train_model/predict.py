"""Make churn predictions with a saved model and report classification metrics."""

import pickle

import pandas as pd
from preprocess import enc_load, load_split_data, model_dir, preprocess_data
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)

from settings import DATASET_NUM, DEBUG, MODEL_DIR, TARGET  # isort:skip


def print_results(classifier, y_test, y_pred, verbose=DEBUG):
    """Print accuracy and balanced accuracy; with verbose, also the full report."""
    print(f'\n================\n{classifier}')
    if verbose:
        print(classification_report(y_test, y_pred, digits=3))
        print(confusion_matrix(y_test, y_pred))
    print(f' accuracy: {accuracy_score(y_test, y_pred):05.3f}')
    print(f' balanced accuracy: {balanced_accuracy_score(y_test, y_pred):05.3f}')


def load_model(model_path):
    """Load the pickled model (a fitted GridSearchCV) from a model folder."""
    with open(f'{model_path}model.pkl', 'rb') as f:
        return pickle.load(f)


def predict_df(df, model_path, verbose=DEBUG):
    """Predict churn (0 or 1) for every row of a preprocessed DataFrame.

    Args:
        df: preprocessed DataFrame (see preprocess.preprocess_data). A TARGET
            column, if present, is ignored for prediction.
        model_path: folder containing model.pkl, ending with '/'.
        verbose: print the first predictions.

    Returns:
        Array of predicted labels, or an empty Series if the model cannot be
        loaded or fails to predict.
    """
    print(f'\nPredicting using model {model_path}')
    X = df.drop(columns=[TARGET], errors='ignore')

    try:
        model = load_model(model_path)
    except (OSError, pickle.UnpicklingError) as e:
        print('!!! Could not load model:', e)
        return pd.Series(dtype=int)

    try:
        # Put the columns in the order the model was trained with.
        if hasattr(model, 'feature_names_in_'):
            X = X[list(model.feature_names_in_)]
        y_pred = model.predict(X)
    except (KeyError, ValueError) as e:  # e.g. a column missing from the input
        print(f'!!! Prediction of {TARGET} failed:', e)
        return pd.Series(dtype=int)

    if verbose:
        print(f'y_pred: {list(y_pred)[:10]}')
        if TARGET in df.columns:
            print(f'y_true: {list(df[TARGET])[:10]}')
    return y_pred


def predict_dict(record, dataset_num, verbose=False):
    """Predict churn for a single customer sent as a dict of raw field values.

    Returns:
        {'churn': 0 or 1}.
    """
    df = pd.DataFrame({key: [value] for key, value in record.items()})
    ord_enc = enc_load(f'{model_dir(dataset_num)}encoder.pkl')
    df = preprocess_data(df, ord_enc, dataset_num, fit_enc=False)
    y_pred = predict_df(df, model_dir(dataset_num), verbose=verbose)
    result = {TARGET.lower(): int(y_pred[0])}  # int() makes it JSON-serialisable
    if DEBUG:
        print('result:', result)
    return result


if __name__ == '__main__':
    # Quick manual check on the held-out test split.
    _, test_df = load_split_data(DATASET_NUM)
    sample = test_df.head(10)
    predictions = predict_df(sample, MODEL_DIR, verbose=True)
    print_results('Saved model', sample[TARGET], predictions, verbose=True)
