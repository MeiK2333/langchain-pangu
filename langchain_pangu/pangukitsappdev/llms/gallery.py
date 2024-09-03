#  Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
import json
import logging
from json import JSONDecodeError
from typing import List, Optional, Iterator, Any

import requests
import sseclient
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.chat_models.base import BaseChatModel
from langchain.llms.base import LLM, BaseLLM
from langchain.schema import AIMessage, BaseMessage
from langchain.schema.messages import AIMessageChunk
from langchain.schema.output import GenerationChunk, LLMResult, Generation, ChatGeneration, ChatResult, \
    ChatGenerationChunk
from requests.exceptions import ChunkedEncodingError

from langchain_pangu.pangukitsappdev.api.common_config import AUTH_TOKEN_HEADER
from langchain_pangu.pangukitsappdev.api.llms.base import AbstractLLMApi, get_llm_params, convert_message_to_req
from langchain_pangu.pangukitsappdev.api.llms.llm_config import LLMConfig
from langchain_pangu.pangukitsappdev.auth.iam import IAMTokenProvider, IAMTokenProviderFactory
from langchain_pangu.pangukitsappdev.llms.response.gallery_text_resp import GalleryUsage, GalleryTextResp, \
    GalleryTextChoice
from langchain_pangu.pangukitsappdev.llms.response.llm_response_gallery import LLMRespGallery

logger = logging.getLogger(__name__)


class GalleryLLM(LLM):
    temperature: Optional[float]
    max_tokens: Optional[int]
    top_p: Optional[float]
    gallery_url: str
    token_getter: IAMTokenProvider
    streaming: Optional[bool]
    proxies: dict = {}

    def _call(self, prompt: str, stop: Optional[List[str]] = None,
              run_manager: Optional[CallbackManagerForLLMRun] = None) -> str:
        llm_result = self._generate([prompt], stop, run_manager)
        return llm_result.generations[0][0].text

    @property
    def _llm_type(self) -> str:
        return "gallery_llm"

    def _stream(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        request_body = {
            "prompt": prompt,
            "stream": True,
            **get_llm_params({
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "top_p": self.top_p
            })
        }
        token = self.token_getter.get_valid_token()
        headers = {
            AUTH_TOKEN_HEADER: token,
            "X-Agent": "pangu-kits-app-dev"
        } if token else {"X-Agent": "pangu-kits-app-dev"}
        rsp = requests.post(self.gallery_url, headers=headers,
                            json=request_body,
                            verify=False, stream=True, proxies=self.proxies)
        try:
            rsp.raise_for_status()
            stream_client: sseclient.SSEClient = sseclient.SSEClient(rsp)
            for event in stream_client.events():
                # 解析出Token数据
                data_json = json.loads(event.data)
                if data_json.get("choices") is None:
                    raise ValueError(f"Meet json decode error: {str(data_json)}, not get choices")
                chunk = GenerationChunk(text=data_json["choices"][0]["text"])
                yield chunk
                if run_manager:
                    run_manager.on_llm_new_token(chunk.text)
        except JSONDecodeError as ex:
            # [DONE]表示stream结束了
            if event.data != "[DONE]":
                logger.warning("Meet json decode error: %s", str(ex))
        except ChunkedEncodingError as ex:
            logger.warning("Meet error: %s", str(ex))

    def _generate(
            self,
            prompts: List[str],
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> LLMResult:
        """Run the LLM on the given prompt and input."""
        generations = []
        llm_output = {}
        for prompt in prompts:
            request_body = {
                "prompt": prompt,
                "stream": False,
                **get_llm_params({
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                    "top_p": self.top_p
                })
            }
            token = self.token_getter.get_valid_token()
            headers = {
                AUTH_TOKEN_HEADER: token,
                "X-Agent": "pangu-kits-app-dev"
            } if token else {"X-Agent": "pangu-kits-app-dev"}
            rsp = requests.post(self.gallery_url, headers=headers,
                                json=request_body,
                                verify=False, stream=False, proxies=self.proxies)

            if 200 == rsp.status_code:
                llm_output = rsp.json()
                text = llm_output["choices"][0]["text"]
                generations.append([Generation(text=text)])
            else:
                logger.error("Call gallery llm failed, http status: %d, error response: %s",
                             rsp.status_code, rsp.content)
                rsp.raise_for_status()

        return LLMResult(generations=generations, llm_output=llm_output)


class GalleryChatLLM(BaseChatModel):
    temperature: Optional[float]
    max_tokens: Optional[int]
    top_p: Optional[float]
    gallery_url: str
    token_getter: IAMTokenProvider
    streaming: Optional[bool]
    proxies: dict = {}

    @property
    def _llm_type(self) -> str:
        return "gallery_chat_llm"

    def _combine_llm_outputs(self, llm_outputs: List[Optional[dict]]) -> dict:
        return llm_outputs[0]

    def _stream(
            self,
            messages: List[BaseMessage],
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        history = self._get_history(messages)
        request_body = {
            "prompt": messages[-1].content,
            "stream": True,
            **get_llm_params({
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "top_p": self.top_p
            }),
            "history": history
        }
        token = self.token_getter.get_valid_token()
        headers = {
            AUTH_TOKEN_HEADER: token,
            "X-Agent": "pangu-kits-app-dev"
        } if token else {"X-Agent": "pangu-kits-app-dev"}
        logger.info("req: %s", json.dumps(request_body, ensure_ascii=False))
        rsp = requests.post(self.gallery_url, headers=headers,
                            json=request_body,
                            verify=False, stream=True, proxies=self.proxies)
        try:
            rsp.raise_for_status()
            stream_client: sseclient.SSEClient = sseclient.SSEClient(rsp)
            for event in stream_client.events():
                # 解析出Token数据
                data_json = json.loads(event.data)
                if data_json.get("choices") is None:
                    raise ValueError(f"Meet json decode error: {str(data_json)}, not get choices")
                chunk = ChatGenerationChunk(message=
                                            AIMessageChunk(content=data_json["choices"][0]["text"]))
                yield chunk
                if run_manager:
                    run_manager.on_llm_new_token(chunk.text)
        except JSONDecodeError as ex:
            # [DONE]表示stream结束了
            if event.data != "[DONE]":
                logger.warning("Meet json decode error: %s", str(ex))
        except ChunkedEncodingError as ex:
            logger.warning("Meet error: %s", str(ex))

    def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None,
                  run_manager: Optional[CallbackManagerForLLMRun] = None, **kwargs: Any) -> ChatResult:
        """Run the LLM on the given prompt and input."""
        history = self._get_history(messages)
        generations = []
        llm_output = {}
        request_body = {
            "prompt": messages[-1].content,
            "stream": False,
            **get_llm_params({
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "top_p": self.top_p
            }),
            "history": history
        }
        token = self.token_getter.get_valid_token()
        headers = {
            AUTH_TOKEN_HEADER: token,
            "X-Agent": "pangu-kits-app-dev"
        } if token else {"X-Agent": "pangu-kits-app-dev"}
        logger.info("req: %s", json.dumps(request_body, ensure_ascii=False))
        rsp = requests.post(self.gallery_url, headers=headers,
                            json=request_body,
                            verify=False, stream=False, proxies=self.proxies)

        if 200 == rsp.status_code:
            llm_output = rsp.json()
            text = llm_output["choices"][0]["text"]
            generations.append(ChatGeneration(message=AIMessage(content=text)))
        else:
            logger.error("Call gallery llm failed, http status: %d, error response: %s", rsp.status_code, rsp.content)
            rsp.raise_for_status()

        return ChatResult(generations=generations, llm_output=llm_output)

    @staticmethod
    def _get_history(messages: List[BaseMessage]) -> List:
        # 第一个role开始作为问答对开始
        message_req = convert_message_to_req(messages)
        while message_req and message_req[0].get("role") != "user":
            messages.pop(0)

        if not message_req:
            raise ValueError("No input messages!")
        history = []
        for i in range(0, len(message_req) - 1, 2):
            if message_req[i].get("role") != "user" and message_req[i+1].get("role") != "assistant":
                raise ValueError("Illegal question and answer pair!")
            history.append([message_req[i].get("content"), message_req[i+1].get("content")])
        return history


class GalleryLLMApi(AbstractLLMApi):

    def do_create_chat_llm(self, llm_config: LLMConfig) -> BaseChatModel:
        config_params = self._parse_llm_config(llm_config)
        return GalleryChatLLM(**config_params)

    def do_create_llm(self, llm_config: LLMConfig) -> BaseLLM:
        config_params = self._parse_llm_config(llm_config)
        return GalleryLLM(**config_params)

    def parse_llm_response(self, llm_result: LLMResult) -> LLMRespGallery:
        answer = llm_result.generations[0][0].text
        llm_output = llm_result.llm_output
        choices = [GalleryTextChoice(index=choice["index"],
                                     text=choice["text"]) for choice in llm_output.get("choices")]
        if llm_output.get("usage") is not None:
            usage = GalleryUsage(completion_tokens=llm_output.get("usage").get("completion_tokens"),
                                 prompt_tokens=llm_output.get("usage").get("prompt_tokens"),
                                 total_tokens=llm_output.get("usage").get("total_tokens"))
        else:
            usage = None

        llm_resp = LLMRespGallery(answer=answer,
                                  is_from_cache=False,
                                  gallery_text_resp=GalleryTextResp(id=llm_output.get("id"),
                                                                    created=llm_output.get("created"),
                                                                    choices=choices,
                                                                    usage=usage))
        return llm_resp

    def _parse_llm_config(self, llm_config: LLMConfig):
        config_params = {
            "gallery_url": llm_config.gallery_config.gallery_url,
            "token_getter": IAMTokenProviderFactory.create(llm_config.gallery_config.iam_config),
            "proxies": llm_config.gallery_config.http_config.requests_proxies()
        }
        llm_param_config_dict: dict = llm_config.llm_param_config.dict(exclude_none=True, exclude={"stream"})
        config_params = {**config_params, **llm_param_config_dict}
        return config_params
