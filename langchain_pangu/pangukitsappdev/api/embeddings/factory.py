#  Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
from typing import Dict, Type

from langchain_pangu.pangukitsappdev.api.common_config import IAMConfigWrapper, HttpConfig
from langchain_pangu.pangukitsappdev.api.embeddings.base import EmbeddingApi
from langchain_pangu.pangukitsappdev.api.embeddings.embedding_config import EmbeddingConfig
from langchain_pangu.pangukitsappdev.embeddings.openai import OpenAIEmbeddingApi
from langchain_pangu.pangukitsappdev.embeddings.pangu import CSSEmbeddingApi
from langchain_pangu.pangukitsappdev.embeddings.tool import ToolEmbeddingApi


class Embeddings:
    embeddings_map: Dict[str, Type[EmbeddingApi]] = {}

    @classmethod
    def register(cls, embedding_type: Type[EmbeddingApi], embedding_name: str):
        """
        注册一种embedding的类型
        :param embedding_type: embedding的类型，要求是EmbeddingApi的子类
        :param embedding_name: embedding的名字，唯一代表这个embedding的名字
        :return: none
        """
        cls.embeddings_map[embedding_name] = embedding_type

    @classmethod
    def of(cls, embedding_name: str, embedding_config: EmbeddingConfig = None) -> EmbeddingApi:
        """
        根据名字创建一个EmbeddingApi的实现类
        :param embedding_name: embedding的名字，唯一标识一种Embedding
        :param embedding_config: （Optional）Embedding的相关配置，如果不传递则从默认配置文件中或者环境变量中获取
        :return: EmbeddingApi
        """

        embedding_type = cls.embeddings_map.get(embedding_name)
        if not embedding_type:
            raise ValueError(
                f"Unregistered embedding name: {embedding_name}, "
                f"please call register(embedding_type, embedding_name) before use.")

        local_embedding_config = embedding_config if embedding_config else cls._load_embedding_config(embedding_name)

        return embedding_type(local_embedding_config)

    @classmethod
    def _load_embedding_config(cls, embedding_name: str):
        if embedding_name == "tool":
            return EmbeddingConfig(env_prefix="sdk.embedding.tool",
                                   iam_config=IAMConfigWrapper(env_prefix="sdk.embedding.tool.iam").get_iam_config(),
                                   http_config=HttpConfig(env_prefix="sdk.embedding.tool.proxy"))
        else:
            return EmbeddingConfig()


Embeddings.register(CSSEmbeddingApi, "css")
Embeddings.register(OpenAIEmbeddingApi, "openAI")
Embeddings.register(ToolEmbeddingApi, "tool")
