"""Project-wide settings for training, prediction and the web service.

Every module imports its configuration from here, so a value changes in one
place only. Directory paths end with '/' because they are joined to file
names by simple string concatenation.
"""

import os

# Print extra diagnostic output. Enable with the environment variable DEBUG=1.
DEBUG = os.getenv('DEBUG', '0') == '1'

# Which churn dataset to use: 1 = e-commerce, 2 = telco.
DATASET_NUM = 1

DATA_DIR = './data/'
MODEL_DIR = f'./model/{DATASET_NUM}/'
MODEL_PREFIX = MODEL_DIR.removeprefix('./')  # S3 key prefix, e.g. 'model/1/'
VISUALS_DIR = './screenshots/'

# Name of the label column (1 = customer churned, 0 = stayed).
TARGET = "Churn"

# Preprocessing
REMOVE_OUTLIERS = False
TEST_SIZE = 0.2  # share of rows held out for the final evaluation
RANDOM_STATE = 42

# MLflow experiment names
EXPERIMENT_NAME = "Training classification models for churn prediction"
HPO_EXPERIMENT_NAME = "Testing trained classification models"

# Web service
PORT = 5555
# Requests with at most this many fields come from dataset 2 (telco);
# larger requests come from dataset 1 (e-commerce).
NEW_API_PARAM_NUM = 13
