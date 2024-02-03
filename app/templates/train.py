from abc import ABC, abstractmethod
import os
import shutil
import subprocess
import time
import traceback
from typing import Text, Union

import boto3
import botocore.client

try:
    import yaml

    load_yaml = yaml.safe_load
    dump_yaml = yaml.safe_dump
except ImportError:
    from ruamel.yaml import YAML

    parser = YAML(typ='safe')
    load_yaml = parser.load
    dump_yaml = parser.dump

LOCAL_VOLUME_PATH = os.getenv('LOCAL_VOLUME_PATH')

S3_ACCESS_KEY_ID = os.getenv('S3_ACCESS_KEY_ID')
S3_SECRET_ACCESS_KEY = os.getenv('S3_SECRET_ACCESS_KEY')
S3_BUCKET_NAME = os.getenv('S3_BUCKET_NAME', 'chatwoot-rasa3')
S3_ENDPOINT = os.getenv('S3_ENDPOINT')
S3_REGION = os.getenv('S3_REGION')

S3_MODEL_DIR = os.getenv('S3_MODEL_DIR', 'model')
S3_CACHE_PATH = os.getenv('S3_CACHE_PATH')
RASA_EXTRA_ARGS = os.getenv('RASA_EXTRA_ARGS')

IS_RASA_FOR_BOTFRONT = os.getenv('IS_RASA_FOR_BOTFRONT')

TRAIN_DATA_DIR = 'data'


class BaseStorage(ABC):
    @abstractmethod
    def upload_file(self, local_path: Text, remote_path: Text) -> None: ...

    @abstractmethod
    def download_file(self, remote_path: Text, local_path: Text) -> None: ...


class LocalStorage(BaseStorage):
    def __init__(self, base_path: Union[Text, None] = LOCAL_VOLUME_PATH) -> None:
        self.base_path = base_path or '.'

    def upload_file(self, local_path: Text, remote_path: Text) -> None:
        r_file_path = f'{self.base_path}/{remote_path}'
        r_file_dir = os.path.dirname(r_file_path)
        os.makedirs(r_file_dir, exist_ok=True)
        shutil.copy(local_path, r_file_path)

    def download_file(self, remote_path: Text, local_path: Text) -> None:
        shutil.copy(f'{self.base_path}/{remote_path}', local_path)


class S3Storage(BaseStorage):
    def __init__(
        self, s3_client: Union[botocore.client.BaseClient, None] = None
    ) -> None:
        self.s3_client = s3_client if s3_client is not None else make_s3_client()

    def upload_file(self, local_path: Text, remote_path: Text) -> None:
        upload_to_s3(self.s3_client, local_path, remote_path)

    def download_file(self, remote_path: Text, local_path: Text) -> None:
        self.s3_client.download_file(S3_BUCKET_NAME, remote_path, local_path)


def main() -> None:

    os.mkdir(TRAIN_DATA_DIR)
    train_data_path = f'{TRAIN_DATA_DIR}/train_data.yml'
    model_path = '/app/model.tar.gz'
    os.mkdir('.rasa')
    local_cache_dir = '.rasa/cache'

    storage: BaseStorage
    if LOCAL_VOLUME_PATH:
        storage = LocalStorage(LOCAL_VOLUME_PATH)
    else:
        storage = S3Storage()

    remote_train_data_path = f'{S3_MODEL_DIR}/train_data.yml'
    print(f'Downloading train data {remote_train_data_path}')
    storage.download_file(remote_train_data_path, train_data_path)
    print(f'Train data downloaded')

    if S3_CACHE_PATH:
        try:
            print(f'Trying to download cache {S3_CACHE_PATH}')
            storage.download_file(S3_CACHE_PATH, 'cache.tar.gz')
            print('Cache downloaded')
            print('Extracting cache')
            shutil.unpack_archive('cache.tar.gz', local_cache_dir)
        except Exception:
            print(f'Error downloading {S3_CACHE_PATH}')
            traceback.print_exc()

    cmd = [
        'rasa',
        'train',
        '-c',
        train_data_path,
        '-d',
        train_data_path,
        '--fixed-model-name',
        model_path,
    ]
    if RASA_EXTRA_ARGS:
        cmd.extend(arg for arg in RASA_EXTRA_ARGS.split(' ') if arg)
    print(f'Running {cmd}', flush=True)
    res = subprocess.run(cmd)
    if res.returncode != 0:
        exit(res.returncode)

    if not os.path.exists(model_path):
        # rasa-for-botfront
        model_path = f'{model_path}.tar.gz'

    storage.upload_file(model_path, f'{S3_MODEL_DIR}/model.tar.gz')
    if S3_CACHE_PATH and os.path.exists(local_cache_dir):
        shutil.make_archive('cache', 'gztar', local_cache_dir)
        storage.upload_file('cache.tar.gz', S3_CACHE_PATH)


def make_s3_client() -> botocore.client.BaseClient:
    session = boto3.Session(
        S3_ACCESS_KEY_ID, S3_SECRET_ACCESS_KEY, region_name=S3_REGION
    )
    return session.client('s3', endpoint_url=S3_ENDPOINT)


def upload_to_s3(
    s3: botocore.client.BaseClient,
    local_path: Text,
    s3_path: Text,
    pause=10,
    max_pause=2560,
) -> None:
    while True:
        print(f'Uploading {local_path} to {s3_path}')
        try:
            s3.upload_file(local_path, S3_BUCKET_NAME, s3_path)
            print(f'{local_path} uploaded')
            break
        except Exception:
            traceback.print_exc()
            time.sleep(pause)
            if pause < max_pause:
                pause *= 2


if __name__ == '__main__':
    main()
