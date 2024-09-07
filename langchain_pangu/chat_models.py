import json
import logging
from json import JSONDecodeError
from typing import (
    List,
    Optional,
    Any,
    Union,
    Dict,
    Type,
    Sequence,
    Callable,
    Iterator,
    AsyncIterator,
)

import aiohttp
import requests
import sseclient
from langchain_core.callbacks import (
    CallbackManagerForLLMRun,
    AsyncCallbackManagerForLLMRun,
)
from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    SystemMessage,
    AIMessage,
    AIMessageChunk,
)
from langchain_core.outputs import (
    ChatResult,
    ChatGenerationChunk,
    ChatGeneration,
)
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from requests.exceptions import ChunkedEncodingError

from langchain_pangu.pangukitsappdev.api.common_config import AUTH_TOKEN_HEADER
from langchain_pangu.pangukitsappdev.api.llms.base import (
    get_llm_params,
    ConversationMessage,
    Role,
)
from langchain_pangu.pangukitsappdev.api.llms.llm_config import LLMConfig
from langchain_pangu.pangukitsappdev.api.tool.base import AbstractTool
from langchain_pangu.pangukitsappdev.auth.iam import (
    IAMTokenProvider,
    IAMTokenProviderFactory,
)
from langchain_pangu.tool_calls import PanguToolCalls


def _pangu_messages(messages: List[BaseMessage]):
    pangu_messages = []
    for message in messages:
        if isinstance(message, SystemMessage):
            # 此处存疑：盘古的 system 看起来效果并不明显
            pangu_messages.append({"role": "system", "content": message.content})
        elif isinstance(message, HumanMessage):
            pangu_messages.append({"role": "user", "content": message.content})
        elif isinstance(message, AIMessage):
            pangu_messages.append({"role": "assistant", "content": message.content})
        else:
            raise ValueError("Received unsupported message type for Pangu.")
    return pangu_messages


class ChatPanGu(BaseChatModel):
    temperature: Optional[float]
    max_tokens: Optional[int]
    top_p: Optional[float]
    presence_penalty: Optional[float]
    llm_config: LLMConfig
    streaming: Optional[bool]
    proxies: dict = {}
    pangu_url: Optional[str]
    token_getter: Optional[IAMTokenProvider]
    with_prompt: Optional[bool]
    tool_calls: Optional[PanguToolCalls]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pangu_url: str = self.llm_config.llm_module_config.url
        self.token_getter = IAMTokenProviderFactory.create(self.llm_config.iam_config)
        self.tool_calls = PanguToolCalls(self.llm_config)

    def _request_body_with_prompt(self, prompt: str):
        return {
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            **get_llm_params(
                {
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                    "top_p": self.top_p,
                    "presence_penalty": self.presence_penalty,
                    "with_prompt": True,
                }
            ),
        }

    def _request_body(self, messages: List[BaseMessage], stream=True):
        rsp = {
            "messages": _pangu_messages(messages),
            "stream": stream,
            **get_llm_params(
                {
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                    "top_p": self.top_p,
                    "presence_penalty": self.presence_penalty,
                    "with_prompt": self.with_prompt,
                }
            ),
        }
        return rsp

    def _headers(self):
        token = self.token_getter.get_valid_token()
        headers = (
            {AUTH_TOKEN_HEADER: token, "X-Agent": "pangu-kits-app-dev"}
            if token
            else {"X-Agent": "pangu-kits-app-dev"}
        )
        return headers

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        proto = self.pangu_url.split("://")[0]
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.pangu_url + "/chat/completions",
                headers=self._headers(),
                json=self._request_body(messages),
                verify_ssl=False,
                proxy=self.proxies[proto] if proto in self.proxies else None,
            ) as rsp:
                while not rsp.closed:
                    line = await rsp.content.readline()
                    if line.startswith(b"data:[DONE]"):
                        rsp.close()
                        break
                    if line.startswith(b"data:"):
                        data_json = json.loads(line[5:])
                        if data_json.get("choices") is None:
                            raise ValueError(
                                f"Meet json decode error: {str(data_json)}, not get choices"
                            )
                        chunk = ChatGenerationChunk(
                            message=AIMessageChunk(
                                content=data_json["choices"][0]["message"]["content"]
                            )
                        )
                        yield chunk
                        if run_manager:
                            await run_manager.on_llm_new_token(chunk.text, chunk=chunk)
                    if line.startswith(b"event:"):
                        pass

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        rsp = requests.post(
            self.pangu_url + "/chat/completions",
            headers=self._headers(),
            json=self._request_body(messages),
            verify=False,
            stream=True,
            proxies=self.proxies,
        )
        try:
            rsp.raise_for_status()
            stream_client: sseclient.SSEClient = sseclient.SSEClient(rsp)
            for event in stream_client.events():
                # 解析出Token数据
                data_json = json.loads(event.data)
                if data_json.get("choices") is None:
                    raise ValueError(
                        f"Meet json decode error: {str(data_json)}, not get choices"
                    )
                chunk = ChatGenerationChunk(
                    message=AIMessageChunk(
                        content=data_json["choices"][0]["message"]["content"]
                    )
                )
                yield chunk
                if run_manager:
                    run_manager.on_llm_new_token(chunk.text, chunk=chunk)
        except JSONDecodeError as ex:
            # [DONE]表示stream结束了
            pass
        except ChunkedEncodingError as ex:
            logging.warning(f"Meet error: %s", str(ex))

    @property
    def _llm_type(self) -> str:
        return "pangu_llm"

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        if "tools" in kwargs:
            prompt = self.tool_calls.tool_calls_prompt(messages)
            body = self._request_body_with_prompt(prompt)
        else:
            body = self._request_body(messages, stream=False)
        proto = self.pangu_url.split("://")[0]
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.pangu_url + "/chat/completions",
                headers=self._headers(),
                json=body,
                verify_ssl=False,
                proxy=self.proxies[proto] if proto in self.proxies else None,
            ) as rsp:
                if rsp.status == 200:
                    llm_output = await rsp.json()
                    text = llm_output["choices"][0]["message"]["content"]
                else:
                    raise ValueError(
                        "Call pangu llm failed, http status: %d",
                        rsp.status,
                    )

        message = AIMessage(
            content=text,
        )
        if "tools" in kwargs:
            message.tool_calls = self.tool_calls.tool_calls(text)
        chat_generation = ChatGeneration(
            message=message,
            generation_info=llm_output,
        )
        return ChatResult(generations=[chat_generation])

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        if "tools" in kwargs:
            prompt = self.tool_calls.tool_calls_prompt(messages)
            body = self._request_body_with_prompt(prompt)
        else:
            body = self._request_body(messages, stream=False)
        rsp = requests.post(
            self.pangu_url + "/chat/completions",
            headers=self._headers(),
            json=body,
            verify=False,
            stream=False,
            proxies=self.proxies,
        )

        if 200 == rsp.status_code:
            llm_output = rsp.json()
            text = llm_output["choices"][0]["message"]["content"]
        else:
            raise ValueError(
                "Call pangu llm failed, http status: %d, error response: %s",
                rsp.status_code,
                rsp.content,
            )

        message = AIMessage(
            content=text,
        )
        if "tools" in kwargs:
            message.tool_calls = self.tool_calls.tool_calls(text)
        chat_generation = ChatGeneration(
            message=message,
            generation_info=llm_output,
        )
        return ChatResult(generations=[chat_generation])

    def bind_tools(
        self,
        tools: Sequence[Union[Dict[str, Any], Type, Callable, BaseTool, AbstractTool]],
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, BaseMessage]:
        for tool in tools:
            self.tool_calls.add_tool(tool)
        return super().bind(tools=tools, **kwargs)

    @staticmethod
    def _message_role(message: BaseMessage):
        if isinstance(message, SystemMessage):
            role = Role.SYSTEM
        elif isinstance(message, HumanMessage):
            role = Role.USER
        elif isinstance(message, AIMessage):
            role = Role.ASSISTANT
        else:
            role = Role.USER
        return role
