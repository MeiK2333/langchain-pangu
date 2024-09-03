#  Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
from langchain.embeddings.base import Embeddings
from langchain_openai import OpenAIEmbeddings

from langchain_pangu.pangukitsappdev.api.embeddings.base import AbstractEmbeddingApi
from langchain_pangu.pangukitsappdev.api.embeddings.embedding_config import EmbeddingConfig


class OpenAIEmbeddingApi(AbstractEmbeddingApi):
    def create_embeddings(self, embedding_config: EmbeddingConfig) -> Embeddings:
        config_params = {}

        if embedding_config.openai_config.openai_base_url:
            config_params["openai_api_base"] = embedding_config.openai_config.openai_base_url

        if embedding_config.openai_config.openai_key:
            config_params["openai_api_key"] = embedding_config.openai_config.openai_key

        # 配置代理
        if embedding_config.openai_config.http_config.proxy_enabled:
            config_params["openai_proxy"] = embedding_config.openai_config.http_config.get_proxy_url()

        return OpenAIEmbeddings(**config_params)
