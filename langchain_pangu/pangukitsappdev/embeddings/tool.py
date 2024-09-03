#  Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
from typing import List

import requests
from langchain.schema.embeddings import Embeddings
from pydantic.v1 import BaseModel

from langchain_pangu.pangukitsappdev.api.embeddings.base import AbstractEmbeddingApi
from langchain_pangu.pangukitsappdev.api.embeddings.embedding_config import EmbeddingConfig


class ToolEmbeddings(Embeddings, BaseModel):
    embedding_config: EmbeddingConfig

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.create_embedding(texts)

    def embed_query(self, text: str) -> List[float]:
        """对query词做Embedding

        检索时对查询词做Embedding

        :param text: 查询词
        :return: :class:`List[float] <List[float]>` 向量数据
        :rtype: List[float]
        """
        return self.embed_documents([text])[0]

    def create_embedding(self, texts: List[str]) -> List[List[float]]:
        proxies = self.embedding_config.http_config.requests_proxies()
        request_body = {
            "content": texts,
            "function": "embedding"
        }

        headers = {"Content-Type": "application/json"}

        rsp = requests.post(url=self.embedding_config.css_url,
                            headers=headers,
                            json=request_body,
                            proxies=proxies,
                            verify=False)
        result = [[]]
        if 200 == rsp.status_code:
            result = rsp.json().get('content')
        else:
            rsp.raise_for_status()
        return result


class ToolEmbeddingApi(AbstractEmbeddingApi):

    def create_embeddings(self, embedding_config: EmbeddingConfig) -> Embeddings:
        return ToolEmbeddings(embedding_config=embedding_config)
