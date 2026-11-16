"""Helpers for storing models in S3-compatible object storage (AWS S3 or LocalStack).

Storage is optional: it is enabled only when the environment (or a .env file)
sets S3_ENDPOINT_URL and S3_BUCKET. When it is enabled, the module exposes a
ready boto3 client as `S3`.
"""

import os
import pickle

from dotenv import load_dotenv

from settings import DEBUG  # isort:skip

load_dotenv()

S3_ENDPOINT_URL = os.getenv("S3_ENDPOINT_URL")
S3_BUCKET = os.getenv("S3_BUCKET")
S3 = None
if S3_ENDPOINT_URL:
    import boto3

    S3 = boto3.client(service_name='s3', endpoint_url=S3_ENDPOINT_URL)
if DEBUG:
    print(f'ENV: S3_ENDPOINT_URL {S3_ENDPOINT_URL}, S3_BUCKET {S3_BUCKET}\n')


def s3_list_buckets(s3):
    """Return the names of all buckets."""
    response = s3.list_buckets()
    return [bucket['Name'] for bucket in response['Buckets']]


def s3_create_bucket(s3, bucket_name):
    """Create a bucket."""
    s3.create_bucket(Bucket=bucket_name)


def s3_list_objects(s3, bucket_name, filter=''):  # pylint: disable=redefined-builtin
    """Return the keys in a bucket, keeping only those that contain `filter`."""
    response = s3.list_objects_v2(Bucket=bucket_name)
    return [obj['Key'] for obj in response.get('Contents', []) if filter in obj['Key']]


def s3_upload_files(s3, bucket_name, local_path, prefix_key='', verbose=DEBUG):
    """Upload a file, or a folder recursively, under the key prefix `prefix_key`."""
    if not os.path.isdir(local_path):
        if verbose:
            print('Uploading:', local_path)
        s3.upload_file(local_path, bucket_name, prefix_key + os.path.basename(local_path))
        return

    for file_name in os.listdir(local_path):
        path = os.path.join(local_path, file_name)
        if os.path.isdir(path):
            s3_upload_files(s3, bucket_name, path, prefix_key + file_name + '/', verbose)
        else:
            if verbose:
                print('Uploading:', path)
            s3.upload_file(path, bucket_name, prefix_key + file_name)


def s3_download_file(s3, bucket_name, object_name, file_name, verbose=DEBUG):
    """Download one object to a local file."""
    if verbose:
        print('Downloading:', object_name)
    s3.download_file(bucket_name, object_name, file_name)


def ensure_bucket(s3, bucket_name):
    """Create `bucket_name` if it does not exist yet."""
    if bucket_name not in s3_list_buckets(s3):
        s3_create_bucket(s3, bucket_name)
        print(f'Created bucket {bucket_name}')


def enc_save(enc, file_name):
    """Save an encoder to a pickle file (same as preprocess.enc_save)."""
    with open(file_name, 'wb') as f:
        pickle.dump(enc, f)


def enc_load(file_name):
    """Load an encoder from a pickle file (same as preprocess.enc_load)."""
    with open(file_name, 'rb') as f:
        return pickle.load(f)


if __name__ == '__main__':
    # Manual check of the storage connection: list what is in the bucket.
    if S3 is None:
        print('S3_ENDPOINT_URL is not set; nothing to check.')
    else:
        ensure_bucket(S3, S3_BUCKET)
        print('Objects:', sorted(s3_list_objects(S3, S3_BUCKET)))
