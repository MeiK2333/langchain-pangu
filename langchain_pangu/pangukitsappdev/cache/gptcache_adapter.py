#  Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
import hashlib
import uuid
from typing import List, Tuple, Dict

import numpy as np
from elasticsearch.helpers import bulk
from gptcache import Cache, Config
from gptcache.adapter.api import init_similar_cache
from gptcache.embedding.langchain import LangChain
from gptcache.manager import get_data_manager
from gptcache.manager.scalar_data.redis_storage import RedisCacheStorage
from gptcache.manager.vector_data.base import VectorBase, VectorData
from langchain.cache import GPTCache

from langchain_pangu.pangukitsappdev.api.memory.cache.base import CacheApiAdapter
from langchain_pangu.pangukitsappdev.api.memory.cache.cache_config import CacheStoreConfig
from langchain_pangu.pangukitsappdev.api.memory.vector.base import Document
from langchain_pangu.pangukitsappdev.api.memory.vector.factory import Vectors
from langchain_pangu.pangukitsappdev.api.memory.vector.vector_config import VectorStoreConfig
from langchain_pangu.pangukitsappdev.vectorstores.bulk_data import BulkData
from langchain_pangu.pangukitsappdev.vectorstores.css_store import CSSVectorSearch

GPT_CACHE_CSS_INDEX_PREFIX = "gptcache_prompt_index"


def text_mapping(dim: int, metric: str = "inner_product") -> Dict:
    return {
        "mappings": {
            "properties": {
                "vector": {
                    "type": "vector",
                    "dimension": dim,
                    "indexing": 'true',
                    "algorithm": "GRAPH",
                    "metric": metric
                },
            }
        }, "settings": {"index": {"vector": "true"}}}


def get_doc_with_score_func(hit: dict) -> Tuple[Document, float]:
    return (
        Document(
            page_content=hit["_id"],
            metadata={},
            id=hit["_id"],
            score=hit["_score"]
        ),
        hit["_score"],
    )


def add_text_request(index_name: str,
                     vector: List[float],
                     content: str,
                     doc_metadata: dict) -> Tuple[str, dict]:
    """这里直接使用content作为es的id"""
    es_id = content
    req = {
        "_op_type": "index",
        "_index": index_name,
        "vector": vector,
        "_id": es_id,
    }

    return es_id, req


class CSSVectorStore(VectorBase):
    """
    适配GPTCache的VectorBase的类，用来集成进gptcache作为向量存储使用

    Attributes:
        _top_k: 默认的4，从CSS中检索出的文档数量
        css_vector_search: 内部分装的CSS向量存储
        vector_config: 用于构造实例的配置内容
    """

    def __init__(self, vector_config: VectorStoreConfig):
        """构造器
        Args:
            vector_config: 向量存储的配置
        """
        config = {
            "elasticsearch_url": vector_config.server_info.get_urls(),
            "index_name": vector_config.index_name,
            "embedding": vector_config.embedding,
            "verify_certs": vector_config.verify_certs,
            "text_key": vector_config.text_key,
            "vector_fields": vector_config.vector_fields,
            "source_fields": vector_config.source_fields,
            "mapping_func": text_mapping,
            "get_doc_with_score_func": get_doc_with_score_func,
        }

        self._top_k = 4
        self.css_vector_search = CSSVectorSearch(**config)
        self.vector_config = vector_config

    def mul_add(self, datas: List[VectorData]):
        bulk_data_list = [BulkData(id=uuid.uuid4(), datas={"texts": str(d.id)}) for d in datas]
        self.css_vector_search.add_docs_with_external_embedding(bulk_data_list=bulk_data_list,
                                                                embedded=[d.data.tolist() for d in datas])

    def search(self, data: np.ndarray, top_k: int):
        """
        相似检索
        Args:
            data: 向量数据
            top_k: 检索top_k个数据

        Returns:
            （score: float, id: int）

        """
        top_k = self._top_k if top_k == -1 else top_k
        docs_and_scores = self.css_vector_search.similarity_search_with_external_embedding(data.tolist(), top_k)
        return [(ds[1], int(ds[0].page_content)) for ds in docs_and_scores]

    def rebuild(self, ids=None) -> bool:
        return True

    def delete(self, ids) -> bool:
        # 构造批量删除请求
        bulk_request = []
        for doc_id in ids:
            delete_request = {
                "_op_type": "delete",  # 操作类型为删除
                "_index": self.vector_config.index_name,  # 要删除的文档所在的索引名称
                "_id": doc_id  # 要删除的文档ID
            }
            bulk_request.append(delete_request)

        # 执行批量删除请求
        bulk(self.css_vector_search.client, bulk_request)

    def flush(self):
        if not self.css_vector_search.client.indices.exists(self.vector_config.index_name):
            return
        self.css_vector_search.client.indices.flush(index=self.vector_config.index_name)


class SemanticGptCacheApi(CacheApiAdapter):
    """GptCache语义缓存
    基于GptCache组件做的适配和扩展。集成了CSS作为vector_store

    """

    def __init__(self, cache_config: CacheStoreConfig):
        def init_gptcache(cache_obj: Cache, llm: str):
            cache = RedisCacheStorage(url=cache_config.server_info.get_urls()[0])
            vector_config = VectorStoreConfig(
                index_name=f"{GPT_CACHE_CSS_INDEX_PREFIX}_{hashlib.sha256(llm.encode()).hexdigest()}",
                vector_fields=["vector"]
            )
            vector = Vectors.of(cache_config.vector_store_name, vector_config)
            data_manager = get_data_manager(cache_base=cache, vector_base=vector)
            init_similar_cache(cache_obj=cache_obj,
                               embedding=LangChain(cache_config.embedding),
                               data_manager=data_manager,
                               config=Config(similarity_threshold=cache_config.score_threshold))

        gpt_cache = GPTCache(init_gptcache)
        super().__init__(gpt_cache, cache_config.session_tag)
