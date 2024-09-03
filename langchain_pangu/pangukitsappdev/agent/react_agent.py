#  Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
from __future__ import unicode_literals

import json
import logging
from json import JSONDecodeError

from pydantic.v1 import ValidationError

from langchain_pangu.pangukitsappdev.agent.agent_action import AgentAction
from langchain_pangu.pangukitsappdev.agent.agent_session import AgentSession
from langchain_pangu.pangukitsappdev.api.agent.base import AbstractAgent, AgentSessionHelper
from langchain_pangu.pangukitsappdev.api.llms.base import LLMApi
from langchain_pangu.pangukitsappdev.prompt.prompt_tmpl import PromptTemplates

logger = logging.getLogger(__name__)


class ReactAgent(AbstractAgent):
    PLACEHOLDER_THOUGHT = "Thought:"
    PLACEHOLDER_ACTION = "Action:"
    PLACEHOLDER_OBSERVATION = "Observation:"
    DEFAULT_SYS_PROMPT = "您是一个智能助手，尽可能有帮助和准确地回答用户提出的问题"

    def __init__(self, llm: LLMApi):
        super(ReactAgent, self).__init__(llm)

    def react(self, agent_session: AgentSession):
        # 超过最大迭代次数限制，不再执行
        if self.need_interrupt(agent_session):
            return

        # 构造React prompt
        react_tp = PromptTemplates.get("agent_react")

        messages = agent_session.messages
        sys_prompt = self.get_system_prompt(
            agent_session) if self.get_system_prompt(agent_session) is not None else self.DEFAULT_SYS_PROMPT
        final_prompt = react_tp.format(sys_prompt=sys_prompt,
                                       tool_desc=self.get_tool_desc(),
                                       tool_names=self.get_tool_names(),
                                       messages=self.convert_message_to_dict(messages))
        # 获取action
        if self.llm.get_llm_config().llm_param_config.stream:
            tokens = self.llm.ask(final_prompt)
            answer = ""
            for token in tokens:
                answer += token
        else:
            answer = self.llm.ask(final_prompt).answer
        action = self.get_action(answer)
        action.req = final_prompt
        action.resp = answer
        agent_session.current_action = action

        # 如果没有结束，执行action
        if action.action != AgentSessionHelper.FINAL_ACTION:
            # 查询tool
            tool = self.get_tool(action.action)
            if tool is None:
                logger.debug("can not find tool for %s in %s", action.action, str(self.tool_map.keys()))
                raise ValueError("agent did not return a valid action")

            # 执行工具
            try:
                self.tool_execute(tool, action.action_input, agent_session)
            except ValidationError:
                logger.debug("agent did not return a valid tool input, input=%s, but tool need %s",
                             action.action_input, tool.input_type)
                raise ValueError("agent did not return a valid tool input")

            logger.info("actions = %s", "\n".join([action.json(ensure_ascii=False)
                                                   for action in agent_session.current_message.actions]))
        else:
            self.notice_session_end(agent_session, action)

    def get_action(self, answer: str) -> AgentAction:
        # 获取第一个action
        action_str = self.sub_str_before(self.sub_str_after(answer, self.PLACEHOLDER_ACTION),
                                         self.PLACEHOLDER_OBSERVATION)
        if not action_str:
            action_str = self.sub_str_before(answer, self.PLACEHOLDER_OBSERVATION)
        agent_action = self.get_agent_action(action_str)
        agent_action.thought = self.remove_start(self.sub_str_before(answer, self.PLACEHOLDER_ACTION),
                                                 self.PLACEHOLDER_THOUGHT).replace("\n", "")
        return agent_action

    @staticmethod
    def get_agent_action(action_str: str) -> AgentAction:
        try:
            action_json = action_str
            agent_action = AgentAction(**json.loads(action_json.replace("actionInput", "action_input")))
        except (JSONDecodeError, TypeError):
            logger.debug("try to load json failed, json:%s", action_json)
            # 尝试通过工程修复一些LLM返回的错误JSON，修正中文逗号
            fix_str = action_str.replace("，", ",")
            try:
                action_json = fix_str
                agent_action = AgentAction(**json.loads(action_json.replace("actionInput", "action_input")))
            except (JSONDecodeError, TypeError):
                logger.debug("After replace the comma, try to load json failed, json:%s", action_json)
                # 查询一个完整的{}，忽略字符串中的{}(简化)
                first_pos = fix_str.find("{")
                if first_pos < 0 or len(fix_str) < 2:
                    raise ValueError("try to fix json failed, agent did not return a valid action")
                match_count = 1
                last_pos = first_pos + 1
                while last_pos < len(fix_str):
                    if fix_str[last_pos] == "{":
                        match_count += 1
                    if fix_str[last_pos] == "}":
                        match_count -= 1
                    if match_count == 0:
                        break
                    last_pos += 1
                try:
                    action_json = fix_str[first_pos: last_pos + 1]
                    agent_action = AgentAction(**json.loads(action_json.replace("actionInput", "action_input")))
                except (JSONDecodeError, TypeError):
                    logger.debug("After fixed json, try to load json failed, json:%s", action_json)
                    raise ValueError("After fixed json, try to load json failed, json:" + action_json)
        agent_action.action_json = action_json
        return agent_action

    def get_tool_desc(self) -> str:
        return PromptTemplates.get("agent_tool_desc").format(tools=[{"toolId": tool.get_tool_id(),
                                                                     "toolDesc": tool.get_tool_desc(),
                                                                     "toolPrinciple": tool.principle,
                                                                     "inputSchema": tool.get_input_schema(),
                                                                     "outputSchema": tool.get_output_schema()}
                                                                    for tool in self.tool_map.values()])

    def get_tool_names(self) -> str:
        return ", ".join(self.tool_map.keys()) + ", " + AgentSessionHelper.FINAL_ACTION
