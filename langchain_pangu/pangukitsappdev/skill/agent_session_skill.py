#  Copyright (c) Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
from pydantic.v1.utils import deepcopy

from langchain_pangu.pangukitsappdev.agent.agent_session import AgentSession
from langchain_pangu.pangukitsappdev.api.llms.base import LLMApi
from langchain_pangu.pangukitsappdev.api.skill.base import Skill, SimpleSkill
from langchain_pangu.pangukitsappdev.prompt.prompt_tmpl import PromptTemplates


class AgentSessionSkill(Skill):
    """
    对Agent的Session进行总结，得出问题的最终答案
    """

    def __init__(self, llm: LLMApi):
        self.llm = llm

    def summary(self, session: AgentSession) -> str:
        history_action = session.current_message.actions
        if not history_action:
            return ""
        # 如果只包含一轮，则直接返回
        if len(history_action) == 1:
            return session.current_message.content
        # 如果包含多轮则进行总结
        summary_skill = SimpleSkill(PromptTemplates.get("skill_agent_session_summary"), self.llm)
        messages = deepcopy(session.messages)
        if messages[-1].role.text == "assistant":
            messages.pop()
        return summary_skill.execute({"messages": session.messages, "actions": history_action})

    def execute(self, session: AgentSession) -> str:
        return self.summary(session)
