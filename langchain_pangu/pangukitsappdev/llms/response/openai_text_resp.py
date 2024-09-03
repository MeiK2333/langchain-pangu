#  Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
from typing import Optional

from pydantic.v1 import BaseModel


class OpenAIUsage(BaseModel):
    """OpenAI模型资源使用情况
    Attributes:
        completion_tokens: 表示模型生成的答案中包含的tokens的数量
        prompt_tokens: 表示生成结果时使用的提示文本的tokens的数量
        total_tokens: 对话过程中使用的tokens总数
    """
    completion_tokens: Optional[int]
    prompt_tokens: Optional[int]
    total_tokens: Optional[int]


class OpenAITextResp(BaseModel):
    """OpenAI模型返回的response
    Attributes:
        model: 模型名
        usage: 模型资源使用情况
    """
    model: Optional[str]
    usage: Optional[OpenAIUsage]
