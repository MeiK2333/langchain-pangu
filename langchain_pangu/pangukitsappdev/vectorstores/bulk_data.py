#  Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
from typing import Dict, Any

from pydantic.v1 import BaseModel, Field


class BulkData(BaseModel):
    """
    ES Bulk消息
    Attributes:
        id: 索引ID
        data: 数据
    """
    id: str
    data: Dict[str, Any] = Field(default={})
