import logging
from typing import Optional, List, Any, Iterator, AsyncIterator

import httpx
from langchain_core.callbacks import (
    CallbackManagerForLLMRun,
    AsyncCallbackManagerForLLMRun,
)
from langchain_core.language_models import LLM
from langchain_core.outputs import GenerationChunk

from langchain_pangu.pangukitsappdev.api.common_config import AUTH_TOKEN_HEADER
from langchain_pangu.pangukitsappdev.api.llms.base import get_llm_params
from langchain_pangu.pangukitsappdev.api.llms.llm_config import LLMConfig
from langchain_pangu.pangukitsappdev.auth.iam import (
    IAMTokenProviderFactory,
    IAMTokenProvider,
)
from langchain_pangu.utils import Utils

logger = logging.getLogger("langchain-pangu")


class PanGuLLM(LLM):
    temperature: Optional[float]
    max_tokens: Optional[int]
    top_p: Optional[float]
    presence_penalty: Optional[float]
    llm_config: LLMConfig
    streaming: Optional[bool]
    proxies: dict = {}
    pangu_url: Optional[str]
    token_getter: Optional[IAMTokenProvider]

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

    def _request_body(self, prompt: str, stream=True):
        rsp = {
            "prompt": prompt,
            "stream": stream,
            **get_llm_params(
                {
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                    "top_p": self.top_p,
                    "presence_penalty": self.presence_penalty,
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

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        resp = httpx.post(
            self.pangu_url + "/text/completions",
            headers=self._headers(),
            json=self._request_body(prompt, stream=False),
            verify=False,
            proxies=self.proxies,
            timeout=None,
        )

        if 200 == resp.status_code:
            llm_output = resp.json()
            text = llm_output["choices"][0]["text"]
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

        return text

    async def _acall(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        async with httpx.AsyncClient(verify=False, proxies=self.proxies) as client:
            resp = await client.post(
                self.pangu_url + "/text/completions",
                headers=self._headers(),
                json=self._request_body(prompt, stream=False),
                timeout=None,
            )

            if resp.status_code == 200:
                llm_output = resp.json()
                text = llm_output["choices"][0]["text"]
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
        return text

    def _stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        with httpx.Client(
            verify=False, proxies=self.proxies, http2=True, http1=False, timeout=None
        ) as client:
            with client.stream(
                "POST",
                self.pangu_url + "/text/completions",
                headers=self._headers(stream=True),
                json=self._request_body(prompt, stream=True),
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
                    chunk = GenerationChunk(text=data["text"])
                    yield chunk
                    if run_manager:
                        run_manager.on_llm_new_token(chunk.text, chunk=chunk)
        yield GenerationChunk(text="")

    async def _astream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[GenerationChunk]:
        async with httpx.AsyncClient(
            verify=False, proxies=self.proxies, http2=True, http1=False, timeout=None
        ) as client:
            async with client.stream(
                "POST",
                self.pangu_url + "/text/completions",
                headers=self._headers(stream=True),
                json=self._request_body(prompt, stream=True),
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
                    chunk = GenerationChunk(text=data["text"])
                    yield chunk
                    if run_manager:
                        await run_manager.on_llm_new_token(chunk.text, chunk=chunk)
        yield GenerationChunk(text="")

    @property
    def _llm_type(self) -> str:
        return "pangu_llm"
