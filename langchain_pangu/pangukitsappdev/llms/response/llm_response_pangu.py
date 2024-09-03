#  Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
from langchain_pangu.pangukitsappdev.api.schema import LLMResp
from langchain_pangu.pangukitsappdev.llms.response.pangu_text_resp import PanguTextResp


class LLMRespPangu(LLMResp):
    """Pangu response封装结构体
    Attributes:
        pangu_text_resp: api返回response结构体
    """
    pangu_text_resp: PanguTextResp

