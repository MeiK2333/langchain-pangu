#  Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
from typing import Dict, Any

from pydantic.v1 import BaseModel, Field


class ToolMetadata(BaseModel):
    """
    工具的元数据信息
    Attributes:
        tool_id: 工具ID
        tool_metadata: 工具元数据，如从ES库中检索到的source字段
    """
    tool_id: str
    tool_metadata: Dict[str, Any] = Field(default={})
