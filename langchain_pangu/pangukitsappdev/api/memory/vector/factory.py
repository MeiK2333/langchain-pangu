#  Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
from typing import Dict, Type

from langchain_pangu.pangukitsappdev.api.memory.vector.base import VectorApi
from langchain_pangu.pangukitsappdev.api.memory.vector.vector_config import VectorStoreConfig
from langchain_pangu.pangukitsappdev.vectorstores.adapter import CSSVectorApi


class Vectors:
    vectors_map: Dict[str, Type[VectorApi]] = {}

    @classmethod
    def register(cls, vector_type: Type[VectorApi], vector_name: str):
        """注册一种vector的类型
        Args:
            vector_type: vector的类型，要求是BaseLLM的子类
            vector_name: vector的名字，唯一代表这个vector的名字
        """
        cls.vectors_map[vector_name] = vector_type

    @classmethod
    def of(cls, vector_name: str = "", vector_config: VectorStoreConfig = None) -> VectorApi:
        """根据名字创建一个VectorApi的实现类
        vector_name和vector_config.store_name不可同时为空
        Args:
            vector_name: （Optional）vector的名字，唯一标识一个VectorApi的实现类
            vector_config: （Optional）VectorApi的相关配置，如果不传递则从默认配置文件中或者环境变量中获取

        Returns:
            一个VectorApi的实现类

        """
        local_vector_name = vector_name if vector_name else vector_config.store_name

        if not local_vector_name:
            raise ValueError("The parameter vector_name or vector_config.store_name should have value")

        vector_type = cls.vectors_map.get(local_vector_name)
        if not vector_type:
            raise ValueError(
                f"Unregistered vector name: {local_vector_name}, \
                please call register(vector_type, vector_name) before use.")

        local_vector_config = vector_config if vector_config else cls._load_config()

        return vector_type(local_vector_config)

    @classmethod
    def _load_config(cls):
        return VectorStoreConfig()


Vectors.register(CSSVectorApi, "css")
