#  Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
from typing import List, Optional

from pydantic.v1 import BaseModel


class GalleryTextChoice(BaseModel):
    """第三方模型生成的补全信息的列表
    Attributes:
        index: 补全的索引，从0开始
        text: 补全的文本内容
    """
    index: Optional[int]
    text: Optional[str]


class GalleryUsage(BaseModel):
    """第三方模型资源使用情况
    Attributes:
        completion_tokens: 表示模型生成的答案中包含的tokens的数量
        prompt_tokens: 表示生成结果时使用的提示文本的tokens的数量
        total_tokens: 对话过程中使用的tokens总数
    """
    completion_tokens: Optional[int]
    prompt_tokens: Optional[int]
    total_tokens: Optional[int]


class GalleryTextResp(BaseModel):
    """第三方模型返回的response
    Attributes:
        id: 用来标识每个响应的唯一字符串
        created: 响应生成的时间
        choices: 生成的补全信息的列表
        usage: 模型资源使用情况
    """
    id: Optional[str]
    created: Optional[str]
    choices: Optional[List[GalleryTextChoice]]
    usage: Optional[GalleryUsage]
