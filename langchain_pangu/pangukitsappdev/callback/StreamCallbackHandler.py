#  Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
from typing import Dict, Any, List

from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import BaseMessage, LLMResult


class StreamCallbackHandler(BaseCallbackHandler):
    def on_chat_model_start(
            self,
            serialized: Dict[str, Any],
            messages: List[List[BaseMessage]],
            **kwargs: Any,
    ) -> Any:
        """
        对话模型开始时运行
        :param serialized: 序列化model
        :param messages: 对话信息
        """

    def on_llm_error(
            self,
            error: BaseException,
            **kwargs: Any,
    ) -> Any:
        """
        对话模型出错时运行
        :param error: 异常信息
        """

    def on_llm_end(
            self,
            response: LLMResult,
            **kwargs: Any,
    ) -> Any:
        """
        对话模型结束时运行
        :param response: 模型结果
        """

    def on_llm_new_token(
            self,
            token: str,
            **kwargs: Any,
    ) -> Any:
        """
        流式每个新token生成时调用
        :param token: 流式生成token
        """
