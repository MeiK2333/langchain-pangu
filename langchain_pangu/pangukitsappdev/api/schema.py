#  Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
from pydantic.v1 import BaseModel


class LLMResp(BaseModel):
    """
    answer: 回答
    is_from_cache: 是否命中缓存
    """

    answer: str
    is_from_cache: bool = False
