#  Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
import logging
from abc import ABC, abstractmethod
from typing import List, Any, Optional, Iterable

from langchain.schema import Document as BaseDocument

try:
    from langchain.vectorstores import VectorStore as BaseVectorStore
except ImportError:
    from langchain.schema.vectorstore import VectorStore as BaseVectorStore
from langchain_pangu.pangukitsappdev.api.memory.vector.vector_config import VectorStoreConfig
from langchain_pangu.pangukitsappdev.vectorstores.bulk_data import BulkData

logger = logging.getLogger(__name__)


class Document(BaseDocument):
    """继承langchain.schema.Document
    增加了打分字段
    Attributes:
        score: 文档的相似度打分
        id: 文档id
    """
    score: float = 0.0
    id: str = ""


class VectorStore(BaseVectorStore, ABC):
    @abstractmethod
    def add_docs(self, bulk_data_list: List[BulkData], **kwargs):
        """写入文档统一接口
        Args:
            bulk_data_list: 批量添加的文档
        """

    def add_texts(self, texts: Iterable[str], metadatas: Optional[List[dict]] = None, **kwargs: Any) -> List[str]:
        raise NotImplementedError("Replace add_texts interface with add_docs interface!")

    @abstractmethod
    def clear(self):
        """清空索引文档"""

    @abstractmethod
    def dsl_search(self, dsl: dict) -> dict:
        """通过DSL查询
        Args:
            dsl: DSL
        Returns:
            原始返回结果
        """


class VectorApi(ABC):
    """Interface for vector store"""
    @abstractmethod
    def add_docs(self, bulk_data_list: List[BulkData], **kwargs):
        """写入文档统一接口
        Args:
            bulk_data_list: 批量添加的文档
        """
        pass

    @abstractmethod
    def similarity_search(self, query: str, top_k: int = 5, score_threshold: float = -1.0) -> List[Document]:
        """相似性检索
        在向量库中检索和query相似的文档
        Args:
            query: 查询文本
            top_k: 返回不超过top_k的document，默认5
            score_threshold: 得分阈值，默认-1.0

        Returns:
            一个Document list

        """
        pass

    @abstractmethod
    def search(self, dsl: str) -> dict:
        """通过DSL查询
        Args:
            dsl: 查询语句
        Returns:
            原始返回结果
        """

    @abstractmethod
    def remove(self, ids: List[str]):
        """删除文档
        Args:
            ids: 需要删除的文档ID
        """
        pass

    @abstractmethod
    def clear(self):
        """清空索引文档"""
        pass


class AbstractVectorApi(VectorApi, ABC):
    """VectorApi接口的基类
    封装了一个langchain.vectorstores.VectorStore的实现类，用来适配VectorApi接口。
    子类需要实现create_vector_store，用来构造一个langchain.vectorstores.VectorStore的实现类
    Attributes:
        vector_config: VectorStoreConfig类型，封装了用来构造VectorStore的一些参数
        vector_store: langchain.vectorstores.VectorStore的实现类，通过create_vector_store方法构造
    """

    def __init__(self, vector_config: VectorStoreConfig):
        self.vector_config = vector_config
        self.vector_store: VectorStore = self.create_vector_store(vector_config)

    def add_docs(self, bulk_data_list: List[BulkData], **kwargs):
        self.vector_store.add_docs(bulk_data_list=bulk_data_list, **kwargs)

    def similarity_search(self, query: str, top_k: int = 5, score_threshold: float = -1.0, **kwargs) -> List[Document]:
        return self.vector_store.similarity_search(query=query, k=top_k, score_threshold=score_threshold)

    def search(self, dsl: dict) -> dict:
        return self.vector_store.dsl_search(dsl)

    def remove(self, ids: List[str]):
        self.vector_store.delete(ids=ids)

    def clear(self):
        self.vector_store.clear()

    @abstractmethod
    def create_vector_store(self, vector_config: VectorStoreConfig) -> VectorStore:
        """使用VectorStoreConfig里的配置构造VectorStore实现类

        Args:
            vector_config: 相关配置

        Returns:
            实现类VectorStore接口的对象
        """
        pass
