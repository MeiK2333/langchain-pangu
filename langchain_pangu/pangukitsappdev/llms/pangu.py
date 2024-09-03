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
from langchain.schema import BaseMessage, AIMessage, get_buffer_string
from langchain.schema.messages import AIMessageChunk
from langchain.schema.output import GenerationChunk, LLMResult, Generation, ChatGenerationChunk, ChatResult, \
    ChatGeneration
from requests.exceptions import ChunkedEncodingError

from langchain_pangu.pangukitsappdev.api.common_config import AUTH_TOKEN_HEADER
from langchain_pangu.pangukitsappdev.api.llms.base import AbstractLLMApi, get_llm_params, convert_message_to_req
from langchain_pangu.pangukitsappdev.api.llms.llm_config import LLMConfig
from langchain_pangu.pangukitsappdev.auth.iam import IAMTokenProvider, IAMTokenProviderFactory
from langchain_pangu.pangukitsappdev.llms.response.llm_response_pangu import LLMRespPangu
from langchain_pangu.pangukitsappdev.llms.response.pangu_text_resp import PanguUsage, PanguTextResp, PanguTextChoice

logger = logging.getLogger(__name__)


class PanguLLM(LLM):
    temperature: Optional[float]
    max_tokens: Optional[int]
    top_p: Optional[float]
    presence_penalty: Optional[float]
    pangu_url: str
    token_getter: IAMTokenProvider
    streaming: Optional[bool]
    proxies: dict = {}

    def _call(self, prompt: str, stop: Optional[List[str]] = None,
              run_manager: Optional[CallbackManagerForLLMRun] = None) -> str:
        llm_result = self._generate([prompt], stop, run_manager)
        return llm_result.generations[0][0].text

    @property
    def _llm_type(self) -> str:
        return "pangu_llm"

    def _stream(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> Iterator[GenerationChunk]:

        messages = [{"role": "user", "content": prompt}]
        request_body = {
            "messages": messages,
            "stream": True,
            **get_llm_params({
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "top_p": self.top_p,
                "presence_penalty": self.presence_penalty
            })
        }

        token = self.token_getter.get_valid_token()
        headers = {
            AUTH_TOKEN_HEADER: token,
            "X-Agent": "pangu-kits-app-dev"
        } if token else {"X-Agent": "pangu-kits-app-dev"}
        rsp = requests.post(self.pangu_url + "/chat/completions", headers=headers,
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
                chunk = GenerationChunk(text=data_json["choices"][0]["message"]["content"])
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
            messages = [{"role": "user", "content": prompt}]
            request_body = {
                "messages": messages,
                "stream": False,
                **get_llm_params({
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                    "top_p": self.top_p,
                    "presence_penalty": self.presence_penalty
                })
            }

            token = self.token_getter.get_valid_token()
            headers = {
                AUTH_TOKEN_HEADER: token,
                "X-Agent": "pangu-kits-app-dev"
            } if token else {"X-Agent": "pangu-kits-app-dev"}
            rsp = requests.post(self.pangu_url + "/chat/completions", headers=headers,
                                json=request_body,
                                verify=False, stream=False, proxies=self.proxies)

            if 200 == rsp.status_code:
                llm_output = rsp.json()
                text = llm_output["choices"][0]["message"]["content"]
                generations.append([Generation(text=text)])
            else:
                logger.error("Call pangu llm failed, http status: %d, error response: %s", rsp.status_code, rsp.content)
                rsp.raise_for_status()

        return LLMResult(generations=generations, llm_output=llm_output)


class PanguChatLLM(BaseChatModel):
    temperature: Optional[float]
    max_tokens: Optional[int]
    top_p: Optional[float]
    presence_penalty: Optional[float]
    pangu_url: str
    token_getter: IAMTokenProvider
    streaming: Optional[bool]
    with_prompt: Optional[bool]
    proxies: dict = {}

    def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None,
                  run_manager: Optional[CallbackManagerForLLMRun] = None, **kwargs: Any) -> ChatResult:
        generations = []
        llm_output = {}
        request_body = {
            "messages": convert_message_to_req(messages),
            "stream": False,
            **get_llm_params({
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "top_p": self.top_p,
                "presence_penalty": self.presence_penalty,
                "with_prompt": self.with_prompt
            })
        }

        token = self.token_getter.get_valid_token()
        headers = {
            AUTH_TOKEN_HEADER: token,
            "X-Agent": "pangu-kits-app-dev"
        } if token else {"X-Agent": "pangu-kits-app-dev"}
        logger.info("req: %s", json.dumps(request_body, ensure_ascii=False))
        rsp = requests.post(self.pangu_url + "/chat/completions", headers=headers,
                            json=request_body,
                            verify=False, stream=False, proxies=self.proxies)

        if 200 == rsp.status_code:
            llm_output = rsp.json()
            text = llm_output["choices"][0]["message"]["content"]
            generations.append(ChatGeneration(message=AIMessage(content=text)))
        else:
            logger.error("Call pangu llm failed, http status: %d, error response: %s", rsp.status_code, rsp.content)
            rsp.raise_for_status()

        return ChatResult(generations=generations, llm_output=llm_output)

    def _stream(
            self,
            messages: List[BaseMessage],
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:

        request_body = {
            "messages": convert_message_to_req(messages),
            "stream": True,
            **get_llm_params({
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "top_p": self.top_p,
                "presence_penalty": self.presence_penalty,
                "with_prompt": self.with_prompt
            })
        }
        token = self.token_getter.get_valid_token()
        headers = {
            AUTH_TOKEN_HEADER: token,
            "X-Agent": "pangu-kits-app-dev"
        } if token else {"X-Agent": "pangu-kits-app-dev"}
        logger.info("req: %s", json.dumps(request_body, ensure_ascii=False))
        rsp = requests.post(self.pangu_url + "/chat/completions", headers=headers,
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
                                            AIMessageChunk(content=data_json["choices"][0]["message"]["content"]))
                yield chunk
                if run_manager:
                    run_manager.on_llm_new_token(chunk.text)
        except JSONDecodeError as ex:
            # [DONE]表示stream结束了
            if event.data != "[DONE]":
                logger.warning("Meet json decode error: %s", str(ex))
        except ChunkedEncodingError as ex:
            logger.warning(f"Meet error: %s", str(ex))

    def _combine_llm_outputs(self, llm_outputs: List[Optional[dict]]) -> dict:
        return llm_outputs[0]

    @property
    def _llm_type(self) -> str:
        return "pangu_chat_llm"

    def get_token_ids(self, text: str) -> List[int]:
        """Get the token present in the text."""
        return self._get_token_ids_default_method(text)

    def get_num_tokens(self, text: str) -> int:
        """Get the number of tokens present in the text."""
        return len(self.get_token_ids(text))

    def get_num_tokens_from_messages(self, messages: List[BaseMessage]) -> int:
        """Get the number of tokens in the message."""
        return sum([self.get_num_tokens(get_buffer_string([m])) for m in messages])

    def _get_token_ids_default_method(self, text: str) -> List[int]:
        """Encode the text into token IDs."""

        request_body = {
            "data": [text],
            "search_result": []
        }
        token = self.token_getter.get_valid_token()
        headers = {
            "x-auth-token": token
        } if token else {}
        rsp = requests.post(self.pangu_url + "/chat/completions", headers=headers,
                            json=request_body,
                            verify=False)

        if 200 == rsp.status_code:
            return rsp.json()["tokens"]

        rsp.raise_for_status()


class PanguLLMApi(AbstractLLMApi):

    def do_create_chat_llm(self, llm_config: LLMConfig) -> BaseChatModel:
        config_params = self._parse_llm_config(llm_config)
        return PanguChatLLM(**config_params)

    def do_create_llm(self, llm_config: LLMConfig) -> BaseLLM:
        config_params = self._parse_llm_config(llm_config)
        return PanguLLM(**config_params)

    def parse_llm_response(self, llm_result: LLMResult) -> LLMRespPangu:
        answer = llm_result.generations[0][0].text
        llm_output = llm_result.llm_output
        choices = [PanguTextChoice(index=choice["index"], text=choice["message"]["content"])
                   for choice in llm_output.get("choices")]
        if llm_output.get("usage") is not None:
            usage = PanguUsage(completion_tokens=llm_output.get("usage").get("completion_tokens"),
                               prompt_tokens=llm_output.get("usage").get("prompt_tokens"),
                               total_tokens=llm_output.get("usage").get("total_tokens"))
        else:
            usage = None
        llm_resp = LLMRespPangu(answer=answer,
                                is_from_cache=False,
                                pangu_text_resp=PanguTextResp(id=llm_output.get("id"),
                                                              created=llm_output.get("created"),
                                                              choices=choices,
                                                              usage=usage))
        return llm_resp

    def _parse_llm_config(self, llm_config: LLMConfig) -> dict:
        config_params = {
            "pangu_url": llm_config.llm_module_config.url,
            "token_getter": IAMTokenProviderFactory.create(llm_config.iam_config),
            "proxies": llm_config.http_config.requests_proxies()
        }
        llm_param_config_dict: dict = llm_config.llm_param_config.dict(exclude_none=True, exclude={"stream"})
        if llm_config.llm_param_config.stream:
            llm_param_config_dict["streaming"] = llm_config.llm_param_config.stream
        config_params = {**config_params, **llm_param_config_dict}
        return config_params
