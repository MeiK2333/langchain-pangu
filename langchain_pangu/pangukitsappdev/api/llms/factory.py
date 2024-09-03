#  Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
from typing import Type, Dict

from langchain_pangu.pangukitsappdev.api.llms.base import LLMApi
from langchain_pangu.pangukitsappdev.api.llms.llm_config import LLMConfig, LLMModuleConfig
from langchain_pangu.pangukitsappdev.llms.gallery import GalleryLLMApi
from langchain_pangu.pangukitsappdev.llms.openai import OpenAILLMApi
from langchain_pangu.pangukitsappdev.llms.pangu import PanguLLMApi


class LLMs:
    llms_map: Dict[str, Type] = {}

    @classmethod
    def register(cls, llm_type: Type[LLMApi], llm_name: str):
        """
        注册一种llm的类型
        :param llm_type: llm的类型，要求是BaseLLM的子类
        :param llm_name: llm的名字，唯一代表这个llm的名字
        :return: none
        """
        cls.llms_map[llm_name] = llm_type

    @classmethod
    def of(cls, llm_name: str, llm_config: LLMConfig = None) -> LLMApi:
        """
        根据名字创建一个LLMApi的实现类
        :param llm_name: llm的名字，唯一标识一种LLM
        :param llm_config: （Optional）LLM的相关配置，如果不传递则从默认配置文件中或者环境变量中获取
        :return: LLMApi
        """

        llm_type = cls.llms_map.get(llm_name)
        if not llm_type:
            raise ValueError(f"Unregistered llm name: {llm_name}, please call register(llm_type, llm_name) before use.")

        local_llm_config = llm_config if llm_config else cls._load_llm_config()

        return llm_type(local_llm_config)

    @classmethod
    def of_module(cls, llm_name: str, llm_module_config: LLMModuleConfig) -> LLMApi:
        """根据指定的LLMModuleConfig构造LLMApi
        Args:
            llm_name:  llm的名字，唯一标识一种LLM
            llm_module_config: 外部参数传递的LLMModuleConfig配置

        Returns: LLMApi

        """
        llm_config = LLMConfig(llm_module_config=llm_module_config)
        return cls.of(llm_name, llm_config)

    @classmethod
    def of_env_prefix(cls, llm_name: str, env_prefix) -> LLMApi:
        """

        Args:
            llm_name: llm的名字，唯一标识一种LLM
            env_prefix: 环境变量或者配置key的前缀

        Returns: LLMApi:

        """
        llm_module_config = LLMModuleConfig(env_prefix=env_prefix)
        return cls.of_module(llm_name, llm_module_config)

    @classmethod
    def _load_llm_config(cls):
        return LLMConfig()


LLMs.register(PanguLLMApi, "pangu")
LLMs.register(OpenAILLMApi, "openAI")
LLMs.register(GalleryLLMApi, "gallery")
