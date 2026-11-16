"""Flask web service that serves churn predictions.

Endpoints (all POST, JSON body):
    /predict      picks the right model from the number of fields and redirects
    /v1/predict   model for dataset 1 (e-commerce)
    /v2/predict   model for dataset 2 (telco)
    /status       model metadata and storage settings
    /update       download the latest model and encoder from S3
    /requestdb    number of stored requests and the time of the latest one

When MONGODB_ADDRESS is set, every request and its prediction is stored in
MongoDB, so the data can later be checked for drift.

The helper modules (preprocess.py, predict.py, settings.py, utils.py) are
copied in from train_model/ when the Docker image is built.
"""

import os

from dotenv import load_dotenv
from flask import Flask, jsonify, redirect, request, url_for
from predict import predict_dict
from pymongo import MongoClient
from pymongo.errors import PyMongoError

from settings import (  # isort:skip
    DATASET_NUM,
    DEBUG,
    MODEL_DIR,
    MODEL_PREFIX,
    NEW_API_PARAM_NUM,
    PORT,
    TARGET,
)
from utils import S3, S3_BUCKET, S3_ENDPOINT_URL, s3_download_file, s3_list_buckets  # isort:skip

load_dotenv()
MONGODB_ADDRESS = os.getenv('MONGODB_ADDRESS', '')

app = Flask('online-prediction')


def get_collection(address):
    """Return the MongoDB collection where requests are stored."""
    client = MongoClient(address)
    return client.get_database('prediction_service').get_collection('data')


collection = get_collection(MONGODB_ADDRESS) if MONGODB_ADDRESS else None


@app.route('/requestdb', methods=['POST'])
def request_mongodb():
    """Report how many requests are stored and when the latest one arrived."""
    address = os.getenv('MONGODB_ADDRESS', 'mongodb://mongo:27017')
    try:
        db_collection = get_collection(address)
        docs_num = db_collection.count_documents({})
        last = list(db_collection.find().sort('_id', -1).limit(1))
    except PyMongoError as e:
        print('MongoDB error:', e)
        return 'MongoDB is not available', 400

    if DEBUG:
        print('MongoDB docs num:', docs_num)
    last_time = str(last[0]['_id'].generation_time) if last else None
    return jsonify({'MONGODB': int(docs_num), 'last_doc': last_time})


def get_model_info(model_path=MODEL_DIR):
    """Read a few metadata fields from the model's MLmodel file."""
    fields = ['mlflow_version', 'model_size_bytes', 'model_uuid', 'run_id', 'utc_time_created']
    model_info = {}
    try:
        with open(f'{model_path}MLmodel', 'r', encoding='utf-8') as f:
            lines = [line.rstrip('\n') for line in f]
    except OSError as e:
        print('Could not read MLmodel:', e)
        return model_info

    for field in fields:
        matches = [line for line in lines if line.startswith(field)]
        if matches:
            model_info[field] = matches[0].split(': ')[-1].strip("'")
    return model_info


@app.route('/status', methods=['POST'])
def service_status():
    """Return model metadata and the storage settings in use."""
    model_info = get_model_info()
    model_info.update(
        {
            'DATASET_NUM': int(DATASET_NUM),
            'MODEL_DIR': MODEL_DIR,
            'S3_ENDPOINT_URL': S3_ENDPOINT_URL,
            'S3_BUCKET': S3_BUCKET,
            'MONGODB_ADDRESS': MONGODB_ADDRESS,
        }
    )
    return jsonify(model_info)


@app.route('/update', methods=['POST'])
def update_model():
    """Download the latest model, encoder and MLmodel file from S3."""
    if S3 is None:
        return 'Empty S3 settings, update failed', 400
    if S3_BUCKET not in s3_list_buckets(S3):
        return f'S3 bucket ({S3_BUCKET}) not found, update failed', 400

    try:
        for file_name in ['encoder.pkl', 'model.pkl', 'MLmodel']:
            s3_download_file(S3, S3_BUCKET, MODEL_PREFIX + file_name, MODEL_DIR + file_name)
    except Exception as e:  # pylint: disable=broad-except
        # boto3 raises several unrelated error types; report any of them to the caller.
        return f'Downloading update from S3 failed: {e}', 500
    return jsonify({'updated': int(DATASET_NUM)})


def save_to_db(record, prediction):
    """Store a request together with its prediction, for drift monitoring."""
    doc = dict(record, prediction=prediction)
    collection.insert_one(doc)


def calculate_prediction(record, dataset_num):
    """Predict with the model for `dataset_num`, falling back to the other model.

    The fallback covers requests sent to the wrong endpoint: there are two
    datasets, so the other one is 3 - dataset_num.
    """
    try:
        result = predict_dict(record, dataset_num, verbose=DEBUG)
    except Exception as e:  # pylint: disable=broad-except
        # Preprocessing and the model raise many error types for mismatched input.
        other = 3 - dataset_num
        print(f'! {e}\nTrying model {other} instead of {dataset_num}')
        try:
            result = predict_dict(record, other, verbose=DEBUG)
        except Exception as e2:  # pylint: disable=broad-except
            print('!! Error:', e2)
            return 'Wrong data/dataset not recognized', 400

    if collection is not None:
        try:
            save_to_db(record, result[TARGET.lower()])
        except PyMongoError as e:
            print('MongoDB error:', e)
    return jsonify(result)


@app.route('/v1/predict', methods=['POST'])
def predict_v1():
    """Predict with the dataset 1 (e-commerce) model."""
    return calculate_prediction(request.get_json(), 1)


@app.route('/v2/predict', methods=['POST'])
def predict_v2():
    """Predict with the dataset 2 (telco) model."""
    return calculate_prediction(request.get_json(), 2)


@app.route('/predict', methods=['POST'])
def predict():
    """Redirect to the endpoint whose dataset matches the request's fields."""
    record = request.get_json()
    dataset_num = 2 if len(record) <= NEW_API_PARAM_NUM else 1
    if DEBUG:
        print('Using API', dataset_num)
    # 307 keeps the POST method and body; 302 would turn it into a GET.
    return redirect(url_for(f'predict_v{dataset_num}'), code=307)


if __name__ == '__main__':
    app.run(debug=DEBUG, host='0.0.0.0', port=PORT)
