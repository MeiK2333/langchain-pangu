#  Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
import logging
import os
import tempfile
from typing import Optional, List, Union

from langchain.docstore.document import Document
from langchain.document_loaders import UnstructuredFileLoader
from langchain.document_loaders.base import BaseLoader
from langchain.document_loaders.directory import FILE_LOADER_TYPE
from obs import LogConf, PutObjectHeader
from pydantic.v1 import Field

logger = logging.getLogger(__name__)

supported_files = [".doc", ".docx", ".pdf", ".txt", ".epub", ".eml", ".msg", ".md", ".pptx", ".html"]


class ObsLoader(BaseLoader):
    def load(self) -> List[Document]:
        # OBS本身是没有文件夹的概念的，文件夹实际上是创建了一个大小为0且对象名以“/”结尾的对象，这类对象与其他对象无任何差异，可以进行下载、删除等操作
        return self._load_from_object_paths(paths=self.object_paths)

    obs_server: str = Field(..., env="OBS_SERVER")
    access_key_id: str = Field(..., env="ACCESS_KEY_ID")
    secret_access_key: str = Field(..., env="SECRET_ACCESS_KEY")
    bucket_name: str
    object_paths: Optional[List[str]] = None
    loader_cls: Optional[FILE_LOADER_TYPE] = UnstructuredFileLoader,
    loader_kwargs: Optional[Union[dict, None]] = None,
    log_path: Optional[str] = None

    def _load_from_object_paths(self, paths: List[str]) -> List[Document]:
        """
        加载obs路径所有支持的文档内容，返回文档列表

        Args:
            paths: obs路径列表

        Returns:
            List[Document]: A list of Document objects representing
            the loaded documents.
        """
        docs = []

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = f"{temp_dir}"
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            # 下载对象-文件下载
            for path in paths:
                filename, file_extension = os.path.splitext(path)
                if file_extension.lower() not in supported_files:
                    self._self_defined_load(docs, file_path)
                    break
                file_path = f"{temp_dir}/{filename}{file_extension}"
                try:
                    error_info = 'download file from obs failed.'
                    resp = self.client.getObject(self.bucket_name, path, downloadPath=file_path)
                    if resp.status < 300:
                        logger.info('requestId:', resp.requestId)
                        logger.info('url:', resp.body.url)
                    else:
                        logger.error('errorCode:', resp.errorCode)
                        logger.error('errorMessage:', resp.errorMessage)
                        break
                    error_info = 'load file type failed.'
                    try:
                        import unstructured  # noqa:F401
                    except ImportError:
                        raise ValueError(
                            "unstructured package not found, please install it with "
                            "`pip install unstructured`"
                        )
                    sub_docs = UnstructuredFileLoader(file_path).load()
                    docs.extend(sub_docs)
                except:
                    logger.error(error_info)
                    self._self_defined_load(docs, file_path)
        return docs

    def _self_defined_load(self, docs, file_path):
        error_info = 'file type not supported.try self-defined loader.'
        logger.error(error_info)
        try:
            sub_docs = self.loader_cls(file_path).load()
            docs.extend(sub_docs)
        except:
            error_info = 'self-defined loader failed.'
            logger.error(error_info)

    def __init__(
            self,
            location: str,
            access_key_id: str,
            secret_access_key: str,
            bucket_name: str,
            object_paths: Optional[List[str]] = None,
            obs_server: Optional[str] = None,
            loader_cls: Optional[FILE_LOADER_TYPE] = UnstructuredFileLoader,
            loader_kwargs: Optional[Union[dict, None]] = None,
            log_path: Optional[str] = None
    ):
        """Initialize with necessary components."""
        try:
            import obs
        except ImportError:
            raise ImportError(
                "Could not import obs python package. "
                "Please install it with `pip install esdk-obs-python`."
            )
        if obs_server:
            self.obs_server = obs_server
        else:
            self.obs_server = f"https://obs.{location}.myhuaweicloud.com"
        self.access_key_id = access_key_id
        self.secret_access_key = secret_access_key
        self.location = location
        self.bucket_name = bucket_name
        self.object_paths = object_paths
        if loader_cls:
            self.loader_cls = loader_cls
        try:
            # 创建ObsClient实例
            self.client = obs.ObsClient(
                access_key_id=access_key_id,
                secret_access_key=secret_access_key,
                server=obs_server
            )
            # 日志初始化
            if log_path:
                # 指定日志配置文件路径，初始化ObsClient日志
                self.client.initLog(LogConf(log_path), obs_server + '-obs.log')

            # 桶初始化
            resp = self.client.headBucket(bucket_name)
            if resp.status == 404:
                logger.info('Bucket does not exist')
                # 创建桶 使用除华北-北京一（cn-north-1）以外的其他终端节点（endpoint）创桶时，必须指
                # 定Location，且Location必须与endpoint中的区域一致。桶会创建在location指定的区域。
                resp = self.client.createBucket(bucketName=bucket_name, location=location)
                if resp.status < 300:
                    logger.info('requestId:', resp.requestId)
                else:
                    logger.error('errorCode:', resp.errorCode)
                    logger.error('errorMessage:', resp.errorMessage)

            # 创建桶客户端
            self.bucket_client = self.client.bucketClient(bucket_name)
        except ValueError as e:
            raise ValueError(
                f"Your obs client string is mis-formatted. Got error: {e} "
            )

    def upload_object(self, local_paths: List[str], des_path: Optional[str] = None):
        headers = PutObjectHeader()
        headers.contentType = 'text/plain'
        # 上传本地文件/文件夹到指定桶中，不能超过5g，否则分段上传
        for local_path in local_paths:
            file_path, filename = os.path.split(local_path)
            if des_path:
                file_path = f"{des_path}/{filename}"
            try:
                resp = self.client.putFile(self.bucket_name, file_path, local_path, headers=headers)
                if resp.status < 300:
                    logger.info('requestId:', resp.requestId)
                    logger.info('etag:', resp.body.etag)
                    logger.info('versionId:', resp.body.versionId)
                    logger.info('storageClass:', resp.body.storageClass)
                else:
                    logger.error('errorCode:', resp.errorCode)
                    logger.error('errorMessage:', resp.errorMessage)
            except:
                logger.error("upload file to obs failed.")

    def delete_object(self, path: str):
        try:
            resp = self.client.deleteObject(self.bucket_name, path)

            if resp.status < 300:
                logger.info('requestId:', resp.requestId)
                logger.info('deleteMarker:', resp.body.deleteMarker)
                logger.info('versionId:', resp.body.versionId)
            else:
                logger.error('errorCode:', resp.errorCode)
                logger.error('errorMessage:', resp.errorMessage)
        except:
            logger.error("delete file on obs failed.")

    def close(self):
        # 关闭obsClient
        self.client.close()
