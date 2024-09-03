#  Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
from abc import ABC, abstractmethod
from typing import Dict, List

from langchain.embeddings.base import Embeddings

from langchain_pangu.pangukitsappdev.api.embeddings.embedding_config import EmbeddingConfig


class EmbeddingApi(Embeddings, ABC):
    @abstractmethod
    def embed_qa_documents(self, doc_texts: List[Dict[str, str]], weight: Dict[str, int]) -> List[List[float]]:
        """
        支持对一份数据的不同文本分权重进行Embedding
        :param doc_texts: 批量索引的文本
        :param weight: 权重，key值和doc_text的元素中的key值一致
        :return: embedding结果
        """
        pass


class AbstractEmbeddingApi(EmbeddingApi, ABC):

    def __init__(self, embedding_config: EmbeddingConfig):
        self.embedding_config = embedding_config
        self.embeddings = self.create_embeddings(embedding_config)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.embeddings.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        return self.embeddings.embed_query(text)

    def embed_qa_documents(self, doc_texts: List[Dict[str, str]], weight: Dict[str, int]) -> List[List[float]]:
        # 默认未实现，直接抛异常
        raise NotImplementedError("Unsupported operation!")

    @abstractmethod
    def create_embeddings(self, embedding_config: EmbeddingConfig) -> Embeddings:
        """
        创建embeddings的实现类
        :param embedding_config: 配置
        :return: embeddings的实现类
        """
        pass
