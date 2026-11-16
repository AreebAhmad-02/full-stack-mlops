"""Train and tune churn classifiers with a grid search.

The input DataFrame is already preprocessed (see preprocess.py). It is split
again here into a fitting part and a validation part, used to pick the best
hyperparameters; the final held-out test split never reaches this module.
"""

import pandas as pd
import xgboost as xgb
from predict import print_results
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeClassifier

from settings import DEBUG, RANDOM_STATE, TARGET  # isort:skip

# Hyperparameter grids searched for each supported classifier.
PARAM_GRIDS = {
    'XGBClassifier': {
        'eta': [0.01, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4],  # learning rate
        'gamma': [0, 1, 2, 3],  # minimum loss reduction to split; larger is more conservative
        'max_depth': [2, 3, 4, 5, 6],
        'min_child_weight': [0, 1, 2],
        'max_delta_step': [0, 1, 2, 4],
        'lambda': [0, 1, 2],  # L2 regularisation on leaf weights
    },
    'DecisionTreeClassifier': {
        'max_depth': [3, 5, 7, 10, 20, 30, None],
        'min_samples_leaf': [1, 3, 5],
        'min_samples_split': [2, 6, 10],
    },
    'RandomForestClassifier': {
        'n_estimators': [25, 50, 100, 150, 200],
        'max_features': ['sqrt', 'log2', None],
        'max_depth': [3, 6, 9, None],
        'min_samples_leaf': [1, 3, 5],
        'min_samples_split': [2, 6, 10],
        'max_leaf_nodes': [3, 6, 9],
    },
}


def make_classifier(name, random_state=RANDOM_STATE):
    """Create an untrained classifier by name; raise ValueError for unknown names."""
    if name == 'XGBClassifier':
        return xgb.XGBClassifier(booster='gbtree', random_state=random_state)
    if name == 'DecisionTreeClassifier':
        return DecisionTreeClassifier(random_state=random_state)
    if name == 'RandomForestClassifier':
        return RandomForestClassifier(random_state=random_state)
    raise ValueError(f'Unknown classifier: {name}')


def prepare_training(df, params):
    """Split a preprocessed DataFrame into features and labels for training.

    Args:
        df: preprocessed DataFrame including the TARGET column.
        params: dict with 'test_size' (default 0.2), 'random_state' (default 42)
            and 'balance'. With balance=True, the larger class is cut down so
            both classes have the same number of rows (churners are roughly
            one in four customers, so the raw data is imbalanced).

    Returns:
        (X_train, X_test, y_train, y_test, params); with balance=True, params
        also gets 'min_subset', the number of rows kept per class.
    """
    test_size = params.get('test_size', 0.2)
    random_state = params.get('random_state', RANDOM_STATE)

    if not params.get('balance', False):
        X = df.drop(columns=[TARGET])
        X_train, X_test, y_train, y_test = train_test_split(
            X, df[TARGET], test_size=test_size, random_state=random_state
        )
        return X_train, X_test, y_train, y_test, params

    print('\nBalancing dataset...')
    class0 = df[df[TARGET] == 0].sample(frac=1, random_state=random_state)
    class1 = df[df[TARGET] == 1].sample(frac=1, random_state=random_state)
    min_len = min(len(class0), len(class1))
    params['min_subset'] = min_len

    n_train = int(min_len * (1 - test_size))
    train = pd.concat([class0[:n_train], class1[:n_train]])
    test = pd.concat([class0[n_train:min_len], class1[n_train:min_len]])
    y_train = train.pop(TARGET)
    y_test = test.pop(TARGET)

    if DEBUG:
        print(' by', df[TARGET].value_counts().to_string())
        print(f'Total: {len(df)}, per class: {min_len}, train {len(train)}, test {len(test)}')
    return train, test, y_train, y_test, params


def train_model(df, params, random_state=RANDOM_STATE):
    """Tune one classifier with a grid search and score it on a validation split.

    Args:
        df: preprocessed training DataFrame including the TARGET column.
        params: dict with 'classifier' (a key of PARAM_GRIDS), 'estimator'
            (the scikit-learn scoring name to optimise, e.g. 'balanced_accuracy'),
            'cv' (cross-validation folds), 'test_size' and 'balance'.
        random_state: seed for the split and the classifier.

    Returns:
        (best_classifier, best_params, validation accuracy, validation balanced accuracy).
    """
    X_train, X_val, y_train, y_val, _ = prepare_training(
        df,
        {
            'balance': params['balance'],
            'test_size': params['test_size'],
            'random_state': random_state,
        },
    )
    classifier_name = params['classifier']
    if classifier_name not in PARAM_GRIDS:
        raise ValueError(f'Unknown classifier: {classifier_name}')

    print(f'\nGridSearchCV {classifier_name}...')
    grid_search = GridSearchCV(
        estimator=make_classifier(classifier_name, random_state),
        param_grid=PARAM_GRIDS[classifier_name],
        cv=params['cv'],
        n_jobs=-1,
        verbose=1,
        scoring=params['estimator'],
    )
    grid_search.fit(X_train, y_train)
    best_classifier = grid_search.best_estimator_
    print(f'Best parameters: {grid_search.best_params_}')

    if DEBUG:
        print(f' best cross-validation score: {grid_search.best_score_}')
        importances = dict(zip(X_train.columns, best_classifier.feature_importances_, strict=True))
        ranked = sorted(importances.items(), key=lambda item: item[1], reverse=True)
        print(classifier_name, 'feature importances:', ranked)

    y_pred = best_classifier.predict(X_val)
    print_results(f"{classifier_name}, optimised for {params['estimator']}", y_val, y_pred)
    return (
        best_classifier,
        grid_search.best_params_,
        accuracy_score(y_val, y_pred),
        balanced_accuracy_score(y_val, y_pred),
    )


if __name__ == '__main__':
    # Train one model without MLflow or Prefect, to check the training code alone.
    from time import time

    from preprocess import load_split_data

    train_df, _ = load_split_data()
    start = time()
    train_model(
        train_df,
        {
            'classifier': 'DecisionTreeClassifier',
            'estimator': 'accuracy',
            'cv': 2,
            'test_size': 0.2,
            'balance': False,
        },
    )
    print(f' Finished in {time() - start:.3f} second(s)\n')
