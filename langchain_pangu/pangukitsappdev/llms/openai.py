#  Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
from langchain.chat_models.base import BaseChatModel
from langchain.llms.base import BaseLLM
from langchain.llms.openai import OpenAI
from langchain.schema import LLMResult
from langchain_openai import ChatOpenAI

from langchain_pangu.pangukitsappdev.api.llms.base import AbstractLLMApi
from langchain_pangu.pangukitsappdev.api.llms.llm_config import LLMConfig
from langchain_pangu.pangukitsappdev.llms.response.llm_response_openai import LLMRespOpenAI
from langchain_pangu.pangukitsappdev.llms.response.openai_text_resp import OpenAIUsage, OpenAITextResp


class OpenAILLMApi(AbstractLLMApi):
    def do_create_llm(self, llm_config: LLMConfig) -> BaseLLM:
        config_params = self._parse_llm_config(llm_config)
        return OpenAI(**config_params)

    def do_create_chat_llm(self, llm_config: LLMConfig) -> BaseChatModel:
        config_params = self._parse_llm_config(llm_config)
        return ChatOpenAI(**config_params)

    def parse_llm_response(self, llm_result: LLMResult) -> LLMRespOpenAI:
        answer = llm_result.generations[0][0].text
        llm_output = llm_result.llm_output

        if llm_output.get("token_usage"):
            usage = OpenAIUsage(completion_tokens=llm_output.get("token_usage").get("completion_tokens"),
                                prompt_tokens=llm_output.get("token_usage").get("prompt_tokens"),
                                total_tokens=llm_output.get("token_usage").get("total_tokens"))
        else:
            usage = None
        llm_resp = LLMRespOpenAI(answer=answer,
                                 is_from_cache=False,
                                 openai_text_resp=OpenAITextResp(model=llm_output.get("model_name"),
                                                                 usage=usage))
        return llm_resp

    def _parse_llm_config(self, llm_config: LLMConfig) -> dict:
        config_params = {}

        if llm_config.openai_config.openai_base_url:
            config_params["openai_api_base"] = llm_config.openai_config.openai_base_url

        if llm_config.openai_config.openai_key:
            config_params["openai_api_key"] = llm_config.openai_config.openai_key

        llm_param_config_dict: dict = llm_config.llm_param_config.dict(exclude_none=True, exclude={"stream"})
        if llm_config.llm_param_config.stream:
            llm_param_config_dict["streaming"] = llm_config.llm_param_config.stream
        config_params = {**config_params, **llm_param_config_dict}

        # 配置代理
        if llm_config.openai_config.http_config.proxy_enabled:
            config_params["openai_proxy"] = llm_config.openai_config.http_config.get_proxy_url()
        return config_params
