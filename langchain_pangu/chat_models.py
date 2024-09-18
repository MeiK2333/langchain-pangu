import logging
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

import httpx
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

from langchain_pangu.pangukitsappdev.api.common_config import AUTH_TOKEN_HEADER
from langchain_pangu.pangukitsappdev.api.llms.base import (
    get_llm_params,
    Role,
)
from langchain_pangu.pangukitsappdev.api.llms.llm_config import LLMConfig
from langchain_pangu.pangukitsappdev.api.tool.base import AbstractTool
from langchain_pangu.pangukitsappdev.auth.iam import (
    IAMTokenProvider,
    IAMTokenProviderFactory,
)
from langchain_pangu.tool_calls import PanguToolCalls
from langchain_pangu.utils import Utils

logger = logging.getLogger("langchain-pangu")


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

    def __init__(
        self,
        pangu_url: str = None,
        project: str = None,
        ak: str = None,
        sk: str = None,
        iam_url: str = None,
        domain: str = None,
        user: str = None,
        password: str = None,
        profile_file: str = None,
        model_version: str = None,
        llm_config: LLMConfig = None,
        temperature: float = None,
        max_tokens: int = None,
        presence_penalty: float = None,
        top_p: float = None,
        proxies: dict = None,
        *args,
        **kwargs,
    ):
        Utils.set_kwargs(
            kwargs,
            pangu_url=pangu_url,
            project=project,
            ak=ak,
            sk=sk,
            iam_url=iam_url,
            domain=domain,
            user=user,
            password=password,
            profile_file=profile_file,
            model_version=model_version,
            llm_config=llm_config,
            temperature=temperature,
            max_tokens=max_tokens,
            presence_penalty=presence_penalty,
            top_p=top_p,
            proxies=proxies,
        )
        super().__init__(*args, **kwargs)

        self.pangu_url: str = self.llm_config.llm_module_config.url
        self.token_getter = IAMTokenProviderFactory.create(self.llm_config.iam_config)
        self.tool_calls = PanguToolCalls(self.llm_config)

    @staticmethod
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
            "messages": self._pangu_messages(messages),
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

    def _headers(self, stream: bool = False):
        headers = {
            AUTH_TOKEN_HEADER: self.token_getter.get_valid_token(),
            "X-Agent": "langchain-pangu",
            "User-Agent": "langchain-pangu",
        }
        if stream:
            headers["Accept"] = "text/event-stream"
            headers["Cache-Control"] = "no-store"
        return headers

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        async with httpx.AsyncClient(
            verify=False, proxies=self.proxies, http2=True, http1=False, timeout=None
        ) as client:
            async with client.stream(
                "POST",
                self.pangu_url + "/chat/completions",
                headers=self._headers(stream=True),
                json=self._request_body(messages),
            ) as stream:
                async for line in stream.aiter_lines():
                    evt, data = Utils.sse_event(line)
                    if evt == Utils.SSE_CONTINUE:
                        continue
                    elif evt == Utils.SSE_DONE:
                        await client.aclose()
                        break
                    elif evt == Utils.SSE_EVENT:
                        continue
                    chunk = ChatGenerationChunk(
                        message=AIMessageChunk(content=data["message"]["content"])
                    )
                    yield chunk
                    if run_manager:
                        await run_manager.on_llm_new_token(chunk.text, chunk=chunk)

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        with httpx.Client(
            verify=False, proxies=self.proxies, http2=True, http1=False, timeout=None
        ) as client:
            with client.stream(
                "POST",
                self.pangu_url + "/chat/completions",
                headers=self._headers(stream=True),
                json=self._request_body(messages),
            ) as stream:
                for line in stream.iter_lines():
                    evt, data = Utils.sse_event(line)
                    if evt == Utils.SSE_CONTINUE:
                        continue
                    elif evt == Utils.SSE_DONE:
                        client.close()
                        break
                    elif evt == Utils.SSE_EVENT:
                        continue
                    chunk = ChatGenerationChunk(
                        message=AIMessageChunk(content=data["message"]["content"])
                    )
                    yield chunk
                    if run_manager:
                        run_manager.on_llm_new_token(chunk.text, chunk=chunk)

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

        async with httpx.AsyncClient(verify=False, proxies=self.proxies) as client:
            resp = await client.post(
                self.pangu_url + "/chat/completions",
                headers=self._headers(),
                json=body,
                timeout=None,
            )
            if resp.status_code == 200:
                llm_output = resp.json()
                text = llm_output["choices"][0]["message"]["content"]
            else:
                logger.error(
                    "Call pangu llm failed, http status: %d, error response: %s",
                    resp.status_code,
                    resp.content,
                )
                raise ValueError(
                    "Call pangu llm failed, http status: %d, error response: %s",
                    resp.status_code,
                    resp.content,
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
        resp = httpx.post(
            self.pangu_url + "/chat/completions",
            headers=self._headers(),
            json=body,
            verify=False,
            proxies=self.proxies,
            timeout=None,
        )
        if 200 == resp.status_code:
            llm_output = resp.json()
            text = llm_output["choices"][0]["message"]["content"]
        else:
            logger.error(
                "Call pangu llm failed, http status: %d, error response: %s",
                resp.status_code,
                resp.content,
            )
            raise ValueError(
                "Call pangu llm failed, http status: %d, error response: %s",
                resp.status_code,
                resp.content,
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
