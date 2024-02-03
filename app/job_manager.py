from enum import Enum
from hashlib import md5
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Iterator, List, Text, Tuple
from uuid import uuid4

import boto3
from boto3.resources.base import ServiceResource
from botocore.response import StreamingBody
import kubernetes
import kubernetes.client as k8s_client
import yaml

LOCAL_VOLUME_PATH = os.getenv('LOCAL_VOLUME_PATH')
LOCAL_VOLUME_PVC_NAME = os.getenv('LOCAL_VOLUME_PVC_NAME')

S3_ACCESS_KEY_ID = os.getenv('S3_ACCESS_KEY_ID')
S3_SECRET_ACCESS_KEY = os.getenv('S3_SECRET_ACCESS_KEY')
S3_BUCKET_NAME = os.getenv('S3_BUCKET_NAME')
S3_ENDPOINT = os.getenv('S3_ENDPOINT')
S3_REGION = os.getenv('S3_REGION')
S3_DIR = os.getenv('S3_DIR', 'models')
S3_CACHE_DIR = os.getenv('S3_CACHE_DIR', 'models-cache')

MODELS_DIR = os.getenv('MODELS_DIR', '/models')

BASE_DIR = Path(__file__).parent.resolve()
TEMPLATES_PATH = f'{BASE_DIR}/templates'
JOB_TEMPLATE_PATH = f'{TEMPLATES_PATH}/job.yml'
TRAIN_PY_TEMPLATE_PATH = f'{TEMPLATES_PATH}/train.py'
CM_TEMPLATE_PATH = f'{TEMPLATES_PATH}/cm.yml'

KUBE_CONFIG_PATH = f'{BASE_DIR}/config/kube'
KUBE_NAMESPACE = os.getenv('KUBE_NAMESPACE', 'external-training')
KUBE_DEFAULT_NAME_PREFIX = os.getenv('KUBE_DEFAULT_NAME_PREFIX', 'et')
KUBE_REGCRED_SECRET_NAME = os.getenv('KUBE_REGCRED_SECRET_NAME', 'bet-regcred')
KUBE_S3_SECRET_NAME = os.getenv('KUBE_S3_SECRET_NAME', 'bet-s3-secret')

IS_RASA_FOR_BOTFRONT = os.getenv('IS_RASA_FOR_BOTFRONT')

DEV = os.getenv('DEV')


logger = logging.getLogger(__name__)


def dict_set(d: Dict, path: Text, value: Any) -> None:
    keys = path.split('.')
    current = d
    for k in keys[:-1]:
        try:
            i = int(k)
            current = current[i]
        except ValueError:
            current = current[k]
    k = keys[-1]
    try:
        i = int(k)
        current[i] = value
    except ValueError:
        current[k] = value


def read_file(path: Text) -> Text:
    with open(path, 'r') as f:
        return f.read()


def load_yml(path: Text) -> Dict[Text, Any]:
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def check_already_exists_error(err: kubernetes.utils.FailToCreateError) -> bool:
    exception: k8s_client.ApiException
    for exception in err.api_exceptions:
        if json.loads(exception.body).get('reason') != 'AlreadyExists':
            return False
    return True


class Status(Enum):
    none = 'none'
    training = 'training'
    success = 'success'
    failed = 'failed'


class K8sJobManager:
    def __init__(self) -> None:
        if os.path.exists(KUBE_CONFIG_PATH):
            self.api_client = kubernetes.config.new_client_from_config(KUBE_CONFIG_PATH)
        else:
            kubernetes.config.load_incluster_config()
            self.api_client = k8s_client.ApiClient()

    def kube_apply(
        self, data: Dict[Text, Any], allow_exists: bool = True
    ) -> List | None:
        try:
            return kubernetes.utils.create_from_dict(
                self.api_client, data, namespace=KUBE_NAMESPACE
            )
        except kubernetes.utils.FailToCreateError as e:
            if allow_exists and check_already_exists_error(e):
                kind = data.get('kind')
                name = data.get('metadata', {}).get('name')
                logger.info(f'{kind} "{name}" already exits')
            else:
                raise
        return None

    @staticmethod
    def job_name(job_id: Text) -> Text:
        return f'{KUBE_DEFAULT_NAME_PREFIX}-job-{job_id}'

    @staticmethod
    def s3_model_dir(project_id: Text, data_hash: Text) -> Text:
        return f'{S3_DIR}/{project_id}/{data_hash}'

    def train(
        self,
        project_id: str,
        image: str,
        training_data: str,
        rasa_extra_args: str | None = None,
        node: str | None = None,
        use_cache: bool = True,
        is_rasa_for_botfront = False,
    ) -> Tuple[Text, bool]:
        training_data = yaml.safe_dump(yaml.safe_load(training_data), sort_keys=True)
        td_hash = md5(training_data.encode()).hexdigest()
        job_id = str(uuid4())

        model_dir = self.s3_model_dir(project_id, td_hash)
        train_data_path = f'{model_dir}/train_data.yml'
        logger.info(f'Uploading train data to {training_data}')
        if LOCAL_VOLUME_PATH:
            train_data_full_path = f'{LOCAL_VOLUME_PATH}/{train_data_path}'
            train_data_dir = os.path.dirname(train_data_full_path)
            os.makedirs(train_data_dir, exist_ok=True)
            with open(train_data_full_path, 'w') as f:
                f.write(training_data)
        else:
            s3 = self._get_s3_resource()
            s3.Object(bucket_name=S3_BUCKET_NAME, key=train_data_path).put(
                Body=training_data.encode(), ContentEncoding='utf-8'
            )
        logger.info('Train data uploaded')

        job_name = self.job_name(job_id)
        cm_name = f'{KUBE_DEFAULT_NAME_PREFIX}-cm-{job_id}'
        job = load_yml(JOB_TEMPLATE_PATH)
        job_envs = [
            {'name': 'S3_MODEL_DIR', 'value': self.s3_model_dir(project_id, td_hash)},
            {'name': 'S3_BUCKET_NAME', 'value': S3_BUCKET_NAME},
            {'name': 'S3_ENDPOINT', 'value': S3_ENDPOINT},
            {'name': 'S3_REGION', 'value': S3_REGION},
        ]
        if rasa_extra_args:
            job_envs.append({'name': 'RASA_EXTRA_ARGS', 'value': rasa_extra_args})
        if use_cache:
            job_envs.append(
                {
                    'name': 'S3_CACHE_PATH',
                    'value': f'{S3_CACHE_DIR}/{project_id}/cache.tar.gz',
                }
            )
        if LOCAL_VOLUME_PATH:
            job_envs.append({'name': 'LOCAL_VOLUME_PATH', 'value': LOCAL_VOLUME_PATH})
        if is_rasa_for_botfront or IS_RASA_FOR_BOTFRONT:
            job_envs.append(
                {'name': 'IS_RASA_FOR_BOTFRONT', 'value': IS_RASA_FOR_BOTFRONT}
            )

        dict_set(job, 'metadata.name', job_name)
        dict_set(job, 'metadata.labels.et/project-id', project_id)
        dict_set(job, 'metadata.labels.et/data-hash', td_hash)
        dict_set(job, 'spec.template.spec.containers.0.env', job_envs)
        dict_set(
            job,
            'spec.template.spec.containers.0.envFrom.0.secretRef.name',
            KUBE_S3_SECRET_NAME,
        )
        dict_set(job, 'spec.template.spec.containers.0.image', image)
        dict_set(job, 'spec.template.spec.volumes.0.configMap.name', cm_name)
        dict_set(
            job, 'spec.template.spec.imagePullSecrets.0.name', KUBE_REGCRED_SECRET_NAME
        )
        if node is not None:
            dict_set(job, 'spec.template.spec.nodeName', node)
        if LOCAL_VOLUME_PATH and LOCAL_VOLUME_PVC_NAME:
            volumes: List[Dict[Text, Any]] = job['spec']['template']['spec']['volumes']
            volumes.append(
                {
                    'name': 'train-data',
                    'persistentVolumeClaim': {'claimName': LOCAL_VOLUME_PVC_NAME},
                }
            )
            volume_mounts: List[Dict[Text, Any]] = job['spec']['template']['spec'][
                'containers'
            ][0]['volumeMounts']
            volume_mounts.append({'name': 'train-data', 'mountPath': LOCAL_VOLUME_PATH})

        # logger.debug(yaml.dump(job))
        created_job = kubernetes.utils.create_from_dict(
            self.api_client, job, namespace=KUBE_NAMESPACE
        )
        logger.info(f'Created job {job_name}')
        # logger.debug(job_name)
        job_uid = created_job[0].metadata.uid

        cm = load_yml(CM_TEMPLATE_PATH)
        dict_set(cm, 'metadata.labels.et/job-id', job_id)
        dict_set(cm, 'metadata.labels.et/project-id', project_id)
        dict_set(cm, 'metadata.labels.et/data-hash', td_hash)
        dict_set(cm, 'metadata.ownerReferences.0.name', job_name)
        dict_set(cm, 'metadata.ownerReferences.0.uid', job_uid)
        dict_set(cm, 'metadata.name', cm_name)
        dict_set(cm, 'data.train_py', read_file(TRAIN_PY_TEMPLATE_PATH))
        # logger.debug(yaml.dump(cm))
        created_cm = kubernetes.utils.create_from_dict(
            self.api_client, cm, namespace=KUBE_NAMESPACE
        )
        logger.info(f'Created cm {cm_name}')

        return job_id, True

    def _get_job(self, job_id: Text) -> k8s_client.V1Job | None:
        job_name = self.job_name(job_id)
        api = k8s_client.BatchV1Api(self.api_client)
        try:
            return api.read_namespaced_job(job_name, KUBE_NAMESPACE)
        except k8s_client.ApiException as e:
            if e.status != 404:
                raise
        return None

    def status(self, job_id: Text) -> Status:
        job = self._get_job(job_id)
        if job is None:
            return Status.none
        job_status: k8s_client.V1JobStatus = job.status
        if job_status.succeeded:
            return Status.success
        if job_status.active:
            return Status.training
        if job_status.failed:
            return Status.failed
        return Status.none

    def cancel(self, job_id: Text) -> bool:
        if self.status(job_id) != Status.training:
            return False
        api = k8s_client.BatchV1Api(self.api_client)
        job_name = self.job_name(job_id)
        opts = k8s_client.V1DeleteOptions(propagation_policy='Background')
        try:
            res = api.delete_namespaced_job(job_name, KUBE_NAMESPACE, body=opts)
        except k8s_client.ApiException as e:
            if e.status == 404:
                return False
            raise
        return True

    def logs(self, job_id: Text) -> Text | None:
        job_name = self.job_name(job_id)
        api = k8s_client.CoreV1Api(self.api_client)
        res: k8s_client.V1PodList = api.list_namespaced_pod(
            KUBE_NAMESPACE, label_selector=f'job-name={job_name}'
        )
        if not res.items:
            return None
        pod: k8s_client.V1Pod = res.items[0]
        try:
            return api.read_namespaced_pod_log(pod.metadata.name, KUBE_NAMESPACE)
        except k8s_client.ApiException as e:
            if e.status == 400:
                # Not started yet
                return None
            raise

    def _get_s3_result_dir(self, job_id: Text) -> Text | None:
        job = self._get_job(job_id)
        if not job:
            return None
        job_status: k8s_client.V1JobStatus = job.status
        if not job_status.succeeded:
            return None

        job_meta: k8s_client.V1ObjectMeta = job.metadata
        labels: Dict[str, str] = job_meta.labels
        project_id = labels['et/project-id']
        data_hash = labels['et/data-hash']
        return self.s3_model_dir(project_id, data_hash)

    def result(self, job_id: Text) -> Iterator[bytes] | Path | None:
        result_dir = self._get_s3_result_dir(job_id)
        if result_dir is None:
            return None

        model_path = f'{result_dir}/model.tar.gz'
        if LOCAL_VOLUME_PATH:
            return Path(f'{LOCAL_VOLUME_PATH}/{model_path}')

        s3 = self._get_s3_resource()
        s3_obj = s3.Object(bucket_name=S3_BUCKET_NAME, key=model_path)
        body: StreamingBody = s3_obj.get()['Body']
        return body

    @staticmethod
    def _get_s3_resource() -> ServiceResource:
        return boto3.Session(
            S3_ACCESS_KEY_ID, S3_SECRET_ACCESS_KEY, region_name=S3_REGION
        ).resource('s3', endpoint_url=S3_ENDPOINT)
