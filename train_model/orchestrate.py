"""Training workflow, orchestrated with Prefect and tracked with MLflow.

Steps:
1. Load the dataset and split it into training and held-out test data.
2. Grid-search classifiers on the training data, logging each run to MLflow.
3. Register the best run's model, copy it to MODEL_DIR and optionally upload it to S3.
4. Evaluate the saved model once on the held-out test data.
"""

import pickle
import shutil
import warnings
from time import time

import mlflow
import pandas as pd
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient
from predict import predict_df, print_results
from prefect import flow, task
from preprocess import load_split_data
from utils import S3, S3_BUCKET, ensure_bucket, s3_list_objects, s3_upload_files

from train_model import train_model

from settings import (  # isort:skip
    DATASET_NUM,
    DEBUG,
    EXPERIMENT_NAME,
    MODEL_DIR,
    MODEL_PREFIX,
    TARGET,
    VISUALS_DIR,
)

# Libraries used here emit many deprecation warnings that are not actionable.
warnings.filterwarnings('ignore')

KEY_METRIC = 'test_accuracy_score'  # metric used to pick the best run
REGISTERED_MODEL_NAME = 'churn-prediction'


@task
def run_experiment(df, params, run_name):
    """Train one classifier configuration and log it as an MLflow run."""
    start = time()
    print(f'\nStarting experiment [{run_name}] with params {params}')

    mlflow.autolog()
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(params)
        _, best_params, accuracy, balanced_accuracy = train_model(df, params)
        mlflow.log_params(best_params)
        mlflow.log_metric('test_accuracy_score', accuracy)
        mlflow.log_metric('test_balanced_accuracy_score', balanced_accuracy)
        mlflow.log_metric('duration', time() - start)
        print('model_uri:', f'runs:/{run.info.run_id}/model')

    print(f'Experiment finished in {time() - start:.3f} second(s)\n')


def feature_importances_chart(feature_importances, classifier_name):
    """Save a bar chart of (feature, importance) pairs to VISUALS_DIR."""
    # Plotting libraries are optional (not in requirements.txt), so import them here.
    import matplotlib.pyplot as plt
    import seaborn as sns

    df = (
        pd.DataFrame(feature_importances, columns=['Feature', 'Importance'])
        .set_index('Feature')
        .sort_values('Importance', ascending=False)
    )
    plt.figure()
    sns.barplot(x='Importance', y='Feature', data=df, color='b')
    plt.title(f'Model {DATASET_NUM} {classifier_name} feature importances')
    plt.tight_layout()  # keeps long feature names inside the image
    plt.savefig(f'{VISUALS_DIR}bar-feature_importances_{classifier_name}-{DATASET_NUM}.png')
    plt.close()


def model_features(model_path, feature_names):
    """Plot the feature importances of the saved model in `model_path`."""
    with open(f'{model_path}model.pkl', 'rb') as f:
        model = pickle.load(f)
    estimator = model.best_estimator_
    classifier_name = type(estimator).__name__
    importances = sorted(
        zip(feature_names, estimator.feature_importances_, strict=True),
        key=lambda item: item[1],
        reverse=True,
    )
    print(classifier_name, 'feature importances:', importances)
    feature_importances_chart(importances, classifier_name)


def upload_model_to_s3(model_path):
    """Upload the model folder and its encoder to the S3 bucket."""
    ensure_bucket(S3, S3_BUCKET)
    s3_upload_files(S3, S3_BUCKET, model_path, prefix_key=MODEL_PREFIX)
    s3_upload_files(S3, S3_BUCKET, f'{model_path}encoder.pkl', prefix_key=MODEL_PREFIX)
    if DEBUG:
        print('S3 objects:', sorted(s3_list_objects(S3, S3_BUCKET)))


@task
def run_register_model(model_path):
    """Register the best run for this dataset, copy it to `model_path`, upload it to S3.

    Returns True if a model was registered.
    """
    print('\nRegistering best model...')
    client = MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    best_runs = client.search_runs(
        experiment_ids=experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        filter_string=f'params.dataset="{DATASET_NUM}"',
        max_results=1,
        order_by=[f'metrics.{KEY_METRIC} DESC'],
    )
    if not best_runs:
        print(f'\n!!! No runs found for dataset {DATASET_NUM}')
        return False

    best_run = best_runs[0]
    print(f'\nBest run {best_run.info.run_id}: metrics {best_run.data.metrics}')
    print(f'params: {best_run.data.params}')

    result = mlflow.register_model(f'runs:/{best_run.info.run_id}/model', REGISTERED_MODEL_NAME)
    print(f'Model registered: {result.name}, version {result.version}')

    # Keep a local copy next to the encoder, for the web service and the tests.
    # download_artifacts resolves both file paths and 'models:/' URIs.
    local_copy = mlflow.artifacts.download_artifacts(artifact_uri=result.source)
    shutil.copytree(local_copy, model_path, dirs_exist_ok=True)
    print(f'Model saved to {model_path}')

    if S3 is not None:
        try:
            upload_model_to_s3(model_path)
        except Exception as e:  # pylint: disable=broad-except
            # Upload is optional: a storage problem must not fail the training run.
            print('!!! Uploading model to S3 failed:', e)
    return True


@task
def test_model(test_df, model_path):
    """Score the saved model once on data it has never seen."""
    print(f'\nTesting model {model_path} on {len(test_df)} held-out rows')
    y_pred = predict_df(test_df, model_path)
    print_results('Saved model, held-out test data', test_df[TARGET], y_pred, verbose=True)


@flow
def ml_workflow():
    """Run the full training workflow (see the module docstring)."""
    mlflow.set_experiment(EXPERIMENT_NAME)
    train_df, test_df = load_split_data()

    # Uncomment more classifiers, folds or balancing options to widen the search.
    for classifier in ['DecisionTreeClassifier']:  # 'RandomForestClassifier', 'XGBClassifier'
        for estimator in ['accuracy', 'balanced_accuracy', 'roc_auc']:
            params = {
                'classifier': classifier,
                'estimator': estimator,
                'dataset': DATASET_NUM,
                'cv': 2,
                'test_size': 0.2,
                'balance': False,
            }
            run_experiment(
                train_df, params=params, run_name=f'classifier {classifier}, estimator {estimator}'
            )

    if run_register_model(MODEL_DIR):
        try:
            model_features(MODEL_DIR, train_df.drop(columns=[TARGET]).columns)
        except ImportError as e:
            print('Skipping feature-importance chart (plotting library missing):', e)
        test_model(test_df, MODEL_DIR)


if __name__ == '__main__':
    ml_workflow()
