import json
import logging
from json import JSONDecodeError
from typing import Optional, List, Any, Iterator, AsyncIterator

import aiohttp
import requests
import sseclient
from langchain_core.callbacks import (
    CallbackManagerForLLMRun,
    AsyncCallbackManagerForLLMRun,
)
from langchain_core.language_models import LLM
from langchain_core.outputs import GenerationChunk
from requests.exceptions import ChunkedEncodingError

from langchain_pangu.pangukitsappdev.api.common_config import AUTH_TOKEN_HEADER
from langchain_pangu.pangukitsappdev.api.llms.base import get_llm_params
from langchain_pangu.pangukitsappdev.api.llms.llm_config import LLMConfig
from langchain_pangu.pangukitsappdev.auth.iam import IAMTokenProviderFactory, IAMTokenProvider


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

    def __init__(self, *args, **kwargs):
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

    def _headers(self):
        token = self.token_getter.get_valid_token()
        headers = (
            {AUTH_TOKEN_HEADER: token, "X-Agent": "pangu-kits-app-dev"}
            if token
            else {"X-Agent": "pangu-kits-app-dev"}
        )
        return headers

    def _call(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> str:
        rsp = requests.post(
            self.pangu_url + "/text/completions",
            headers=self._headers(),
            json=self._request_body(prompt, stream=False),
            verify=False,
            stream=False,
            proxies=self.proxies,
        )

        if 200 == rsp.status_code:
            llm_output = rsp.json()
            text = llm_output["choices"][0]["text"]
        else:
            raise ValueError(
                "Call pangu llm failed, http status: %d, error response: %s",
                rsp.status_code,
                rsp.content,
            )

        return text

    async def _acall(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> str:
        proto = self.pangu_url.split("://")[0]
        async with aiohttp.ClientSession() as session:
            async with session.post(
                    self.pangu_url + "/text/completions",
                    headers=self._headers(),
                    json=self._request_body(prompt, stream=False),
                    verify_ssl=False,
                    proxy=self.proxies[proto] if proto in self.proxies else None,
            ) as rsp:
                if rsp.status == 200:
                    llm_output = await rsp.json()
                    text = llm_output["choices"][0]["text"]
                else:
                    raise ValueError(
                        "Call pangu llm failed, http status: %d",
                        rsp.status,
                    )
        return text

    def _stream(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        rsp = requests.post(
            self.pangu_url + "/text/completions",
            headers=self._headers(),
            json=self._request_body(prompt),
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
                chunk = GenerationChunk(text=data_json["choices"][0]["text"])
                yield chunk
                if run_manager:
                    run_manager.on_llm_new_token(chunk.text, chunk=chunk)
        except JSONDecodeError as ex:
            # [DONE]表示stream结束了
            pass
        except ChunkedEncodingError as ex:
            logging.warning(f"Meet error: %s", str(ex))

    async def _astream(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> AsyncIterator[GenerationChunk]:
        proto = self.pangu_url.split("://")[0]
        async with aiohttp.ClientSession() as session:
            async with session.post(
                    self.pangu_url + "/text/completions",
                    headers=self._headers(),
                    json=self._request_body(prompt),
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
                        chunk = GenerationChunk(text=data_json["choices"][0]["text"])
                        yield chunk
                        if run_manager:
                            await run_manager.on_llm_new_token(chunk.text, chunk=chunk)
                    if line.startswith(b"event:"):
                        pass

    @property
    def _llm_type(self) -> str:
        return "pangu_llm"
