#  Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
from abc import ABC, abstractmethod
from typing import Any, Dict, Union, List

from langchain.chains.llm import LLMChain
from langchain.schema.prompt_template import BasePromptTemplate

from langchain_pangu.pangukitsappdev.api.llms.base import LLMApi, ConversationMessage
from langchain_pangu.pangukitsappdev.api.llms.llm_config import LLMParamConfig


class Skill(ABC):

    @abstractmethod
    def execute(self, inputs: Dict[str, Any]) -> str:
        """
        执行Executor
        :param inputs: 输入的参数，基本上都是用来渲染prompt_template的
        :return: 执行结果，LLM的返回结果
        """
        pass

    @staticmethod
    def skill_llm_ask(prompt: Union[str, List[ConversationMessage]],
                      llm_api: LLMApi,
                      param_config: LLMParamConfig = None) -> str:
        """
        支持skill流式输出
        :param prompt: 用户输入
        :param llm_api: 模型api接口
        :param param_config: 可选参数
        :return: 执行结果，LLM的返回结果
        """
        if (param_config and param_config.stream) or llm_api.get_llm_config().llm_param_config.stream:
            tokens = llm_api.ask(prompt, param_config)
            answer = ""
            for token in tokens:
                answer += token
        else:
            answer = llm_api.ask(prompt, param_config).answer
        return answer


class SimpleSkill(Skill):
    """
    一个Executor的简单实现，传递prompt_template和llm_api两个参数即可，面向api包下面的接口编程
    """

    def __init__(self, prompt_template: BasePromptTemplate, llm_api: LLMApi):
        self.prompt_template = prompt_template
        self.llm_api = llm_api

    def execute(self, inputs: Dict[str, Any], param_config: LLMParamConfig = None) -> str:
        prompt = self.prompt_template.format(**inputs)
        return self.skill_llm_ask(prompt, self.llm_api, param_config)


class ChainWrappedSkill(Skill):
    """
    通过封装一个Chain来实现Executor，方便集成langchain预置的一些chain
    """

    def __init__(self, chain: LLMChain):
        self.chain = chain

    def execute(self, inputs: Dict[str, Any]) -> str:
        # 默认返回output_keys的第一个元素所代表的输出内容，一般都是回答
        chain_output_key = self.chain.output_keys[0]

        return self.chain(inputs)[chain_output_key]
