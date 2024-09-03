#  Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
from langchain_pangu.pangukitsappdev.api.schema import LLMResp
from langchain_pangu.pangukitsappdev.llms.response.openai_text_resp import OpenAITextResp


class LLMRespOpenAI(LLMResp):
    """OpenAI response封装结构体
    Attributes:
        openai_text_resp: openai sdk返回结构体
    """
    openai_text_resp: OpenAITextResp

