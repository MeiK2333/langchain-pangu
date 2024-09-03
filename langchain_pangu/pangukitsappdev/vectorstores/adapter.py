#  Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.

try:
    from langchain.vectorstores import VectorStore
except ImportError:
    from langchain.schema.vectorstore import VectorStore

from langchain_pangu.pangukitsappdev.api.memory.vector.base import AbstractVectorApi
from langchain_pangu.pangukitsappdev.api.memory.vector.vector_config import VectorStoreConfig
from langchain_pangu.pangukitsappdev.vectorstores.css_store import CSSVectorSearch


class CSSVectorApi(AbstractVectorApi):
    def create_vector_store(self, vector_config: VectorStoreConfig) -> VectorStore:
        config = {
            "elasticsearch_url": vector_config.server_info.get_urls(),
            "index_name": vector_config.index_name,
            "embedding": vector_config.embedding,
            "verify_certs": vector_config.verify_certs,
            "text_key": vector_config.text_key,
            "vector_fields": vector_config.vector_fields,
            "source_fields": vector_config.source_fields,
            "proxies": vector_config.http_config.requests_proxies()
        }

        return CSSVectorSearch(**config)
