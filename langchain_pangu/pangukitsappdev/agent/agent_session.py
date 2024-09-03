#  Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
from typing import Optional, List

from pydantic.v1 import BaseModel

from langchain_pangu.pangukitsappdev.agent.agent_action import AgentAction
from langchain_pangu.pangukitsappdev.api.llms.base import ConversationMessage


class AgentSession(BaseModel):
    """
    Agent运行Session，包含历史Action，当前Action，状态
    Attributes:
        messages: 本次session的用户的输入
        session_id: UUID，在一个session内唯一
        current_action: 当前Action
        agent_session_status: Agent状态
        is_by_step: 是否是逐步执行
        current_message: 当前的AssistantMessage
    """
    messages: List[ConversationMessage]
    session_id: str
    current_action: Optional[AgentAction]
    agent_session_status: str
    is_by_step: Optional[bool]
    current_message: Optional[ConversationMessage]
