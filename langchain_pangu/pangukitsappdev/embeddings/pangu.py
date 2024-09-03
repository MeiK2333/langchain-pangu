#  Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
from typing import List, Dict

import requests
from langchain.embeddings.base import Embeddings
from pydantic.v1 import BaseModel, Extra

from langchain_pangu.pangukitsappdev.api.common_config import AUTH_TOKEN_HEADER
from langchain_pangu.pangukitsappdev.api.embeddings.base import AbstractEmbeddingApi
from langchain_pangu.pangukitsappdev.api.embeddings.embedding_config import EmbeddingConfig
from langchain_pangu.pangukitsappdev.auth.iam import IAMTokenProvider, IAMTokenProviderFactory


def _default_doc_map(text: str) -> dict:
    """默认文档拆分方法
    封装成{"content": text}返回
    :param text: 文本内容
    :return: :class:`dict` {"content": text}
    :rtype: dict
    """
    return {"content": text}


def json_doc_map(text: str, encoding: str = "utf8") -> Dict[str, str]:
    import json
    return json.loads(text, encoding=encoding)


class WeightPanguEmbeddings(Embeddings, BaseModel):
    """支持文本拆分，不同部分使用不同的权重做Embedding

    """

    url: str
    token_provider: IAMTokenProvider
    proxies: dict = {}

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.ignore
        arbitrary_types_allowed = True

    def auth_headers(self) -> dict:

        token = self.token_provider.get_valid_token()
        headers = {
            AUTH_TOKEN_HEADER: token,
            "X-Agent": "pangu-kits-app-dev"
        } if token else {"X-Agent": "pangu-kits-app-dev"}
        return headers

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """批量对文本做Embedding
        一般用在构造向量数据库的时候批量进行文档Embedding

        :param texts: 文档数据
        :return: :class:`list[list[float]] <list[list[float]]>` 以二维数组格式返回的向量数据
        :rtype: list[list[float]]
        """

        request_body = {
            "query": texts
        }

        headers = self.auth_headers()
        batch_url = f"{self.url}/embedding/query"
        rsp = requests.post(batch_url, headers=headers, json=request_body, proxies=self.proxies, verify=False)

        if 200 == rsp.status_code:
            result = rsp.json()["embedding"]
            return result

        rsp.raise_for_status()

    def embed_qa_documents(self, doc_texts: List[Dict[str, str]], weight: Dict[str, int]) -> List[List[float]]:
        request_body = {
            "weight": weight,
            "docs": doc_texts
        }

        headers = self.auth_headers()

        batch_url = f"{self.url}/embedding/batch"
        rsp = requests.post(batch_url, headers=headers, json=request_body, proxies=self.proxies, verify=False)

        if 200 == rsp.status_code:
            result = rsp.json()["embedding"]
            return result

        rsp.raise_for_status()

    def embed_query(self, text: str) -> List[float]:
        """对query词做Embedding

        检索时对查询词做Embedding

        :param text: 查询词
        :return: :class:`List[float] <List[float]>` 向量数据
        :rtype: List[float]
        """
        return self.embed_documents([text])[0]


    def embed_split_documents(self, documents: List, weight: Dict[str, int]) -> List[List[float]]:
        """
        根据weight指定需要embedding的字段及其权重
        :param documents: 文档列表
        :param weight: 字段权重
        :return:
        """
        doc_texts = []
        # 需要判断field是否是document的属性
        if not weight.keys() <= documents[0].dict().keys():
            raise Exception("weight keys should be in documents")

        for document in documents:
            doc_text = {}
            for field in weight:
                doc_text[field] = getattr(document, field)
            doc_texts.append(doc_text)

        return self.embed_qa_documents(doc_texts, weight)


class CSSEmbeddingApi(AbstractEmbeddingApi):
    """
    CSS的Embedding API，支持embed_qa_documents
    """

    def embed_qa_documents(self, doc_texts: List[Dict[str, str]], weight: Dict[str, int]) -> List[List[float]]:
        return self.embeddings.embed_qa_documents(doc_texts=doc_texts, weight=weight)

    def create_embeddings(self, embedding_config: EmbeddingConfig) -> Embeddings:
        token_provider = IAMTokenProviderFactory.create(embedding_config.iam_config)
        return WeightPanguEmbeddings(url=embedding_config.css_url,
                                     token_provider=token_provider,
                                     proxies=embedding_config.http_config.requests_proxies())

    def embed_split_documents(self, doc_texts: List, weight: Dict[str, int]) -> List[List[float]]:
        return self.embeddings.embed_split_documents(doc_texts, weight)
