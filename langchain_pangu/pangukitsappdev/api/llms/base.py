#  Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
import json
import logging
from abc import ABC, abstractmethod
from enum import Enum
from typing import Iterator, Type, List, Union, Literal, Optional, Any, Dict

from langchain.callbacks.base import BaseCallbackHandler
from langchain.chat_models.base import BaseChatModel
from langchain.schema import LLMResult, BaseMessage
from pydantic.v1 import BaseModel

from langchain_pangu.pangukitsappdev.agent.agent_action import AgentAction
from langchain_pangu.pangukitsappdev.api.llms.llm_config import LLMParamConfig, LLMConfig
from langchain_pangu.pangukitsappdev.api.memory.cache.base import CacheApi
from langchain_pangu.pangukitsappdev.api.schema import LLMResp
from langchain_pangu.pangukitsappdev.prompt.prompt_tmpl import PromptTemplates

logger = logging.getLogger(__name__)


class Role(Enum):
    SYSTEM = {"text": "system", "desc": "系统"}
    USER = {"text": "user", "desc": "用户"}
    ASSISTANT = {"text": "assistant", "desc": "助手"}
    OBSERVATION = {"text": "observation", "desc": "观察"}

    @property
    def desc(self):
        return self.value.get("desc")

    @property
    def text(self):
        return self.value.get("text")


class ConversationMessage(BaseMessage):
    """多轮对话信息
    拓展了tool和actions，可用于Agent使用

    Attributes:
        role: 对话角色
        content: 对话内容
        tools: 工具集
        actions: 当role为assistant时，采取的actions，如果没有采取任何action，则为空数组
    """
    role: Role
    """The speaker / role of the Message."""

    type: Literal["chat"] = "chat"

    tools: Optional[Any]

    actions: List[AgentAction] = []


def convert_message_to_req(messages: List[BaseMessage]) -> List[Dict]:
    req = []
    for message in messages:
        if not isinstance(message, ConversationMessage):
            raise ValueError("Pangu LLM input message must be inherited from ConversationMessage!")
        req.append({"role": message.role.text, "content": message.content})
    return req


class LLMApi(ABC):

    @abstractmethod
    def ask(self,
            prompt: Union[str, List[ConversationMessage]],
            param_config: LLMParamConfig = None) -> Union[LLMResp, Iterator]:
        """
        问答
        :param prompt: 单轮提示词或多轮message list
        :param param_config: (Optional)覆盖llm的原本的参数配置，用来控制llm的返回信息
        :return: LLMResp or Iterator(流式打印)
        """

    @abstractmethod
    def set_cache(self, cache: CacheApi):
        """
        设置缓存
        :param cache: 缓存实现对象
        :return: void
        """

    @abstractmethod
    def set_callback(self, callback: BaseCallbackHandler):
        """
        设置Callback回调对象
        :param callback: callback对象
        """

    @abstractmethod
    def ask_for_object(self, prompt: str, class_type: Type[BaseModel], param_config: LLMParamConfig = None):
        """
        问答
        :param prompt: 提示词
        :param class_type: 需要LLM转换的类型
        :param param_config: (Optional)覆盖llm的原本的参数配置，用来控制llm的返回信息
        :return: LLM answer
        """

    @abstractmethod
    def get_llm_config(self) -> LLMConfig:
        """
        获取当前LLM的配置
        :return: LLMConfig
        """


class AbstractLLMApi(LLMApi):

    @staticmethod
    def _build_llm_string(chat_llm: BaseChatModel) -> str:
        return str(sorted([(k, v) for k, v in chat_llm.dict().items()]))

    @staticmethod
    def _parse_llm_response(llm_result: LLMResult) -> LLMResp:
        answer_generation = llm_result.generations[0][0]
        answer = answer_generation.text
        llm_resp = LLMResp(answer=answer, is_from_cache=False)
        return llm_resp

    def __init__(self, llm_config: LLMConfig, chat_llm: BaseChatModel = None, cache: CacheApi = None):
        """ 构造器

          Args:
              llm_config: 通过构造器传递的LLM参数
              chat_llm: （Optional）内部封装的Langchain BaseChatModel的实现类
              cache: CacheApi的实现类，为LLM增加缓存
        """
        self.llm_config = llm_config
        self.chat_llm = chat_llm if chat_llm else self.create_chat_llm_with()

        self.callback_handler = None
        self.cache = cache

    def ask(self,
            prompt: Union[str, List[ConversationMessage]],
            param_config: LLMParamConfig = None) -> Union[LLMResp, Iterator]:
        # 根据入参LLMParamConfig判断是否是流式打印，返回类型为Iterator[str]
        if param_config and param_config.stream:
            return self._stream(prompt=prompt, param_config=param_config)

        # 根据构造器传递的LLMConfig中的LLMParamConfig判断是否是流式打印
        if self.llm_config and self.llm_config.llm_param_config and self.llm_config.llm_param_config.stream:
            return self._stream(prompt=prompt)

        # 非流式打印返回类型为LLMResp
        local_llm = self.chat_llm if param_config is None else self.create_chat_llm_with(param_config)

        # 尝试从缓存获取响应
        if self.cache and isinstance(prompt, str):
            """移除session_tag的构造，session_tag在cache初始化时传入，不在lookup中传递
            """
            cached_llm_rsp = self.cache.lookup(prompt=prompt)
            if cached_llm_rsp:
                """这里需要复制所有的成员变量，当前只有answer和is_from_cache"""
                logger.debug("Hit cached completion. prompt: %s", prompt)
                return LLMResp(answer=cached_llm_rsp.answer, is_from_cache=True)

            logger.debug("Miss cache")

        llm_result: LLMResult = local_llm.generate([self._get_messages(prompt)],
                                                   callbacks=[self.callback_handler] if self.callback_handler else None)

        llm_resp = self.parse_llm_response(llm_result)
        logger.info("resp: %s", llm_resp.answer)
        # 尝试更新缓存
        if self.cache and not llm_resp.is_from_cache and isinstance(prompt, str):
            self.cache.update(prompt=prompt,
                              value=llm_resp)
            logger.debug("Update cache for prompt %s", prompt)

        return llm_resp

    def ask_for_object(self, prompt: str, class_type: Type[BaseModel], param_config: LLMParamConfig = None):
        final_prompt = PromptTemplates.get("system_out_put_parser").format(schema=dict(
            properties=class_type.schema()["properties"]), prompt=prompt)
        llm_resp = self.ask(final_prompt, param_config)

        # 流式打印拼凑结果后输出
        if isinstance(llm_resp, Iterator):
            actual_tokens = []
            for token in llm_resp:
                actual_tokens.append(token)
            answer = "".join(actual_tokens)
        else:
            answer = llm_resp.answer
        return class_type(**json.loads(answer))

    def set_cache(self, cache: CacheApi):
        self.cache = cache

    def default_create_chat_llm_func(self, param_config: LLMParamConfig) -> BaseChatModel:
        """创建chat_llm的默认方法
        使用chat_llm的类直接构造，调用这个默认实现是self.chat_llm必须存在
        使用param_config构造新的chat_llm
        :param param_config: chat_llm参数配置
        :return: 使用新参数构造的chat_llm
        """
        if not self.chat_llm:
            raise ValueError("the default create chat_llm func need preset a BaseChatModel instance")
        llm_type = type(self.chat_llm)
        return llm_type(**param_config.dict())

    def set_callback(self, callback: BaseCallbackHandler):
        self.callback_handler = callback

    def _stream(self,
                prompt: Union[str, List[ConversationMessage]],
                param_config: LLMParamConfig = None) -> Iterator[str]:
        local_llm = self.chat_llm if param_config is None else self.create_chat_llm_with(param_config)
        # 尝试从缓存获取响应
        if self.cache and isinstance(prompt, str):
            """缓存命中，直接从缓存中读取，返回size为1的迭代器
            """
            cached_llm_rsp = self.cache.lookup(prompt=prompt)
            if cached_llm_rsp:
                """返回命中cache"""
                logger.debug("Hit cached completion. prompt: %s", prompt)
                yield cached_llm_rsp.answer
                return
            logger.debug("Miss cache")
        # 通过stream方式获取response

        tokens = local_llm.stream(self._get_messages(prompt),
                                  config=dict(callbacks=[self.callback_handler]) if self.callback_handler else None)

        answer = ""
        for token in tokens:
            yield token.content
            answer += token.content

        logger.info("resp: %s", answer)
        llm_resp = LLMResp(answer=answer, is_from_cache=False)

        # 尝试更新缓存
        if self.cache and not llm_resp.is_from_cache and isinstance(prompt, str):
            self.cache.update(prompt=prompt,
                              value=llm_resp)
            logger.debug("Update cache for prompt %s", prompt)

    def do_create_chat_llm(self, llm_config: LLMConfig):
        return self.default_create_chat_llm_func(llm_config.llm_param_config)

    def create_chat_llm_with(self, param_config: LLMParamConfig = None) -> BaseChatModel:
        if param_config:
            llm_config: LLMConfig = self.llm_config.copy(deep=True)
            llm_config.llm_param_config = param_config
            return self.do_create_chat_llm(llm_config)
        else:
            return self.do_create_chat_llm(self.llm_config)

    def parse_llm_response(self, llm_result: LLMResult) -> LLMResp:
        return self._parse_llm_response(llm_result)

    def _get_messages(self, prompt: Union[str, List[ConversationMessage]]) -> List[ConversationMessage]:
        messages: List[ConversationMessage] = []
        if self.need_add_new_system_message():
            messages.append(ConversationMessage(role=Role.SYSTEM,
                                                content=self.llm_config.llm_module_config.system_prompt))
        if isinstance(prompt, str):
            messages.append(ConversationMessage(role=Role.USER, content=prompt))
        else:
            for msg in prompt:
                if self.need_add_new_system_message() and msg.role.text == "system":
                    continue
                messages.append(msg)
        return messages

    def get_llm_config(self) -> LLMConfig:
        return self.llm_config

    def need_add_new_system_message(self) -> bool:
        return self.llm_config.llm_module_config.enable_append_system_message and \
            self.llm_config.llm_module_config.system_prompt is not None


def get_llm_params(params: dict) -> dict:
    """
    用于过滤llm api参数
    若配置参数，则返回对应参数字典，否则返回空，用默认参数值
    :param params: llm api支持的参数
    :return: 消息体中的dict数据
    """
    llm_param_config = LLMParamConfig(**params)
    return llm_param_config.dict(exclude_none=True)


class LLMApiAdapter(AbstractLLMApi):
    """LLMApi的适配器
    负责把Langchain的LLM实现类适配到LLMApiAdapter

    Attributes:
        chat_llm: 内部封装的Langchain BaseChatModel的实现类
    """

    def __init__(self, chat_llm: BaseChatModel):
        llm_config = LLMConfig()
        super().__init__(llm_config, chat_llm)
