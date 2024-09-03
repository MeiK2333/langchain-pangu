#  Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
from typing import Any, Optional

from pydantic.v1 import BaseModel


class AgentAction(BaseModel):
    """
    Agent执行的action
    Attributes:
        req: LLM的原始请求
        resp: LLM的原始响应
        thought: LLM的思考
        action_json: LLM输出的原始Action字符串
        action: LLM将要采取的行动，即Tool
        action_input: 行动的输入，即Tool的输入
        observation: 采取行动后给LLM观察的结果，即Tool的执行结果
        user_feedback: 用户对该步骤的反馈，可能需要更改工具调用参数
    """
    req: Optional[str]
    resp: Optional[str]
    thought: Optional[str]
    action_json: Optional[str]
    action: Optional[str]
    action_input: Optional[Any]
    observation: Optional[str]
    user_feedback: Optional[str]
