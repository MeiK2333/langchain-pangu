"""Wrapper around CSS vector database."""
from __future__ import annotations

import logging
import uuid
from abc import ABC
from typing import Any, Dict, List, Optional, Tuple, Callable

from langchain.embeddings.base import Embeddings
from langchain.utils import get_from_env

from langchain_pangu.pangukitsappdev.api.memory.vector.base import VectorStore, Document
from langchain_pangu.pangukitsappdev.utils.time_date import now_millis
from langchain_pangu.pangukitsappdev.vectorstores.bulk_data import BulkData

logger = logging.getLogger(__name__)


def _default_text_mapping(dim: int, metric: str = "inner_product") -> Dict:
    return {
        "mappings": {
            "properties": {
                "text": {"type": "text"},
                "vector": {
                    "type": "vector",
                    "dimension": dim,
                    "indexing": 'true',
                    "algorithm": "GRAPH",
                    "metric": metric
                },
            }
        }, "settings": {"index": {"vector": "true"}}}


def _default_script_query(query_vector: List[float], vector_field: str = "vector",
                          metric: str = "inner_product") -> Dict:
    return {
        "script_score": {
            "query": {"match_all": {}},
            "script": {
                "source": "vector_score",
                "lang": "vector",
                "params": {
                    "field": vector_field,
                    "vector": query_vector,
                    "metric": metric
                }
            }
        }
    }


def _default_get_doc_with_score_func(hit: dict, text_key: str) -> Tuple[Document, float]:
    return (
        Document(
            page_content=str(hit["_source"].get(text_key)),
            metadata=hit["_source"],
            id=hit["_id"],
            score=hit["_score"]
        ),
        hit["_score"],
    )


# CSSVectorSearch是抽象基类VectorStore的具体实现，它为所有矢量数据库实现定义了一个公共接口。
# 通过继承ABC类，弹性矢量搜索可以定义为抽象基类本身，允许创建具有自己特定实现的子类。
class CSSVectorSearch(VectorStore, ABC):
    """把CSS数据库包装成向量数据库，CSS是基于开源Elasticsearch开发的自研向量搜索数据库，连接不需要身份认证信息, 传入 URL 和 index 以及
    embedding 进行构造.
    Args:
        elasticsearch_url (str): The URL for the Elasticsearch instance.
        index_name (str): The name of the Elasticsearch index for the embeddings.
        embedding (Embeddings): An object that provides the ability to embed text.
                It should be an instance of a class that subclasses the Embeddings
                abstract base class, such as OpenAIEmbeddings()

    Raises:
        ValueError: If the elasticsearch python package is not installed.
    """

    def __init__(
            self,
            elasticsearch_url: str,
            index_name: str,
            embedding: Embeddings,
            vector_fields: List[str],
            source_fields: List[str],
            text_key: str,
            mapping_func: Callable[[int], Dict] = _default_text_mapping,
            script_query_func: Callable[[List[float], str], dict] = _default_script_query,
            get_doc_with_score_func: Callable[[Dict, str], Tuple[Document, float]] = _default_get_doc_with_score_func,
            **kwargs
    ):
        """
        初始化
        Args:
            elasticsearch_url: 作为elasticsearch.client.Elasticsearch的hosts参数，支持逗号分割的list
            index_name: 索引名称
            embedding: （Optional）embedding接口的实现类。如果不传递，则走文本检索的逻辑
            mapping_func: （Optional）支持传递自定义的构造mapping函数，允许自定义mapping。入参为向量字段的长度。返回索引的mapping
            script_query_func: （Optional）支持传递自定义的查询dsl构造方法，入参为待搜索的向量vector: List[float]。返回query dsl
            get_doc_with_score_func: (Optional)传递自定义的从检索结果获取（Document, score）数据的函数，入参是es的检索结果
            add_text_request_func: (Optional)传递自定义的函数，在批量添加索引数据时构造请求体。入参：索引名，向量，文档内容，文档元数据\n
                出参（文档id, 构造的请求体）
            **kwargs:
        """

        """Initialize with necessary components."""
        try:
            import elasticsearch
        except ImportError:
            raise ImportError(
                "Could not import elasticsearch python package. "
                "Please install it with `pip install elasticsearch`."
            )
        self.embedding = embedding
        self.index_name = index_name
        self.mapping_func = mapping_func
        self.script_query_func = script_query_func
        self.get_doc_with_score_func = get_doc_with_score_func
        self.vector_fields = vector_fields
        self.source_fields = source_fields
        self.text_key = text_key
        try:
            from langchain_pangu.pangukitsappdev.vectorstores.proxy_http_requests import ProxyRequestsHttpConnection
            self.client = elasticsearch.Elasticsearch(elasticsearch_url, connection_class=ProxyRequestsHttpConnection,
                                                      **kwargs)
        except ValueError as e:
            raise ValueError(
                f"Your elasticsearch client string is mis-formatted. Got error: {e} "
            )

    def add_docs(self, bulk_data_list: List[BulkData], **kwargs):
        batch_size = kwargs.get("batch_size", 100)
        # 一批一批的数据进行索引构建
        sub_bulk_data = []
        for i, d in enumerate(bulk_data_list):
            sub_bulk_data.append(d)
            if (i + 1) % batch_size == 0:
                if self.is_external_embedding():
                    vector_field = self.vector_fields[0]
                    texts = [bulk.data.get(vector_field, "") for bulk in sub_bulk_data]
                    self.add_docs_with_external_embedding(sub_bulk_data,
                                                          self.embedding.embed_documents(texts))
                else:
                    self.add_docs_with_builtin_embedding(sub_bulk_data)
                sub_bulk_data = []
                logger.info("Indexed %s document", str(i+1))

        if sub_bulk_data:
            if self.is_external_embedding():
                vector_field = self.vector_fields[0]
                texts = [bulk.data.get(vector_field, "") for bulk in sub_bulk_data]
                self.add_docs_with_external_embedding(sub_bulk_data,
                                                      self.embedding.embed_documents(texts))
            else:
                self.add_docs_with_builtin_embedding(sub_bulk_data)

    def add_docs_with_external_embedding(self, bulk_data_list: List[BulkData], embedded: List[List[float]]):
        vector_field = self.vector_fields[0]
        bulk_requests = []

        if not self.client.indices.exists(index=self.index_name):
            # 外部embedding模式可自检索引
            if self.mapping_func:
                mappings = self.mapping_func(len(embedded[0]))
                if not self.client.indices.exists(index=self.index_name):
                    # 创建索引和映射
                    self.client.indices.create(index=self.index_name,
                                               body=mappings)
                    logger.info("Index created successfully.")
            else:
                raise ImportError(
                    "index is not exists. Indexes should be created first."
                )
        for i, bulk_data in enumerate(bulk_data_list):
            req = {
                "_op_type": "index",
                "_index": self.index_name,
                "_id": bulk_data.id,
            }
            req.update(bulk_data.data)
            req.update({vector_field: embedded[i]})
            bulk_requests.append(req)
        try:
            from elasticsearch.exceptions import NotFoundError
            from elasticsearch.helpers import bulk
        except ImportError:
            raise ImportError(
                "Could not import elasticsearch python package. "
                "Please install it with `pip install elasticsearch`."
            )
        bulk(self.client, bulk_requests)
        self.client.indices.refresh(index=self.index_name)

    def add_docs_with_builtin_embedding(self, bulk_data_list: List[BulkData]):
        bulk_requests = []
        if not self.client.indices.exists(index=self.index_name):
            # CSS add_docs 索引需外部创建好
            raise ImportError(
                "index is not exists. Indexes should be created first."
            )
        for i, bulk_data in enumerate(bulk_data_list):
            req = {
                "_op_type": "index",
                "_index": self.index_name,
                "_id": bulk_data.id,
            }
            req.update(bulk_data.data)
            bulk_requests.append(req)
        try:
            from elasticsearch.exceptions import NotFoundError
            from elasticsearch.helpers import bulk
        except ImportError:
            raise ImportError(
                "Could not import elasticsearch python package. "
                "Please install it with `pip install elasticsearch`."
            )
        bulk(self.client, bulk_requests)
        self.client.indices.refresh(index=self.index_name)

    def similarity_search(
            self, query: str, k: int = 5,
            score_threshold: float = -1.0, **kwargs: Any
    ) -> List[Document]:
        """返回和查询语句最相近的k条文本.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 5.
            score_threshold: 低于这个阈值分数的doc不会被检索出来
        Returns:
            List of Documents most similar to the query.
        """
        docs_and_scores = self.similarity_search_with_score(query, k, score_threshold, **kwargs)
        documents = [d[0] for d in docs_and_scores]
        return documents

    def similarity_search_with_score(
            self, query: str, k: int = 5,
            score_threshold: float = -1.0,
            **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        """Return docs most similar to query.
        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 5.
            score_threshold: （Optional）低于这个阈值分数的doc不会被检索出来
        Returns:
            List of Documents most similar to the query.
        """

        if self.is_external_embedding():
            embedding = self.embedding.embed_query(query)
            docs_with_score = self.similarity_search_with_external_embedding(embedding, k, **kwargs)
        else:
            # 如果没有传递embedding，则执行如下逻辑
            docs_with_score = self.similarity_search_with_builtin_embedding(query=query, k=k, **kwargs)

        return [dws for dws in docs_with_score if dws[1] >= score_threshold]

    def similarity_search_with_external_embedding(self, embedding, k, **kwargs) -> List[Tuple[Document, float]]:
        vector_field = self.vector_fields[0]
        response = self.client.search(index=self.index_name,
                                      body={
                                          'query': self.script_query_func(embedding, vector_field),
                                          '_source': self.source_fields,
                                          'size': k
                                      }, size=k, **kwargs)
        hits = [hit for hit in response["hits"]["hits"]]
        docs_and_scores = [
            self.get_doc_with_score_func(hit, self.text_key)
            for hit in hits
        ]
        return docs_and_scores

    def similarity_search_with_builtin_embedding(self, query, k, **kwargs) -> List[Tuple[Document, float]]:
        """检索,不使用
        :param query: query词
        :param k: 检索结果数
        :param kwargs: 扩展参数
        :return: :class:`List[Document] <List[Document]>` 检索结果
        """

        query_dsl = {
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": self.vector_fields
                }
            },
            "from": 0,
            "size": k,
            "_source": self.source_fields
        }
        start_millis = now_millis()
        response = self.client.search(index=self.index_name, body=query_dsl, size=k, **kwargs)
        end_millis = now_millis()

        duration_ms = end_millis - start_millis

        # 服务端处理的耗时信息
        server_cost_time = response.get("timestamp")
        # 服务端处理的状态
        server_status = response.get("status")

        logging.info(
            "Success return, request cost: [duration_ms: %s], [server_cost_time: %s], "
            "[server_status: %s]", str(duration_ms), str(server_cost_time), str(server_status))

        hits = [hit for hit in response["hits"]["hits"]]
        # 解析搜索结果，支持从参数中传递解析函数
        docs_and_scores = [
            self.get_doc_with_score_func(hit, self.text_key)
            for hit in hits
        ]
        return docs_and_scores

    def similarity_search_with_relevance_scores(self, query: str, k: int = 5,
                                                score_threshold: float = -1.0,
                                                **kwargs: Any) -> List[Tuple[Document, float]]:
        return self.similarity_search_with_score(query, k, score_threshold, **kwargs)

    def is_external_embedding(self) -> bool:
        if not self.embedding:
            return False
        if len(self.vector_fields) != 1:
            logger.error("add docs with external embedding, "
                         "but the vector field is not config or have multi vector fields")
            raise ValueError("add docs with external embedding, "
                             "but the vector field is not config or have multi vector fields")
        return True

    def dsl_search(self, dsl: dict) -> dict:
        response = self.client.search(index=self.index_name, body=dsl)
        return response

    def delete(self, ids: Optional[List[str]] = None, **kwargs: Any):
        delete_query = {
            "query": {
                "terms": {
                    "_id": ids
                }
            }
        }
        self.client.delete_by_query(index=self.index_name, body=delete_query)

    def clear(self):
        delete_query = {
            "query": {
                "match_all": {}
            }
        }
        self.client.delete_by_query(index=self.index_name, body=delete_query)

    @classmethod
    def from_texts(
            cls,
            texts: List[str],
            embedding: Embeddings,
            metadatas: Optional[List[dict]] = None,
            elasticsearch_url: Optional[str] = None,
            index_name: Optional[str] = None,
            refresh_indices: bool = True,
            **kwargs: Any,
    ) -> CSSVectorSearch:
        """Construct ElasticVectorSearch wrapper from raw documents.

        This is a user-friendly interface that:
            1. Embeds documents.
            2. Creates a new index for the embeddings in the Elasticsearch instance.
            3. Adds the documents to the newly created Elasticsearch index.

        This is intended to be a quick way to get started.

        Example:
            .. code-block:: python

                from langchain import ElasticVectorSearch
                from langchain.embeddings import OpenAIEmbeddings
                embeddings = OpenAIEmbeddings()
                elastic_vector_search = ElasticVectorSearch.from_texts(
                    texts,
                    embeddings,
                    elasticsearch_url="http://localhost:9200"
                )
        """
        elasticsearch_url = elasticsearch_url or get_from_env(
            "elasticsearch_url", "ELASTICSEARCH_URL"
        )
        index_name = index_name or uuid.uuid4().hex
        vector_search = cls(elasticsearch_url, index_name, embedding, **kwargs)
        vector_search.add_texts(
            texts, metadatas=metadatas, refresh_indices=refresh_indices
        )
        return vector_search
