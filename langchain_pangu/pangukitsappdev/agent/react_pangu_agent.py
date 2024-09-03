#  Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
from __future__ import unicode_literals

import json
import logging
from json import JSONDecodeError

from langchain.prompts import PromptTemplate

from langchain_pangu.pangukitsappdev.agent.agent_action import AgentAction
from langchain_pangu.pangukitsappdev.agent.agent_session import AgentSession
from langchain_pangu.pangukitsappdev.api.agent.base import AbstractAgent, AgentSessionHelper
from langchain_pangu.pangukitsappdev.api.llms.base import LLMApi
from langchain_pangu.pangukitsappdev.prompt.prompt_tmpl import PromptTemplates

logger = logging.getLogger(__name__)


class ReactPanguAgent(AbstractAgent):
    DEFAULT_SYS_PROMPT = "你的名字叫智子，是由华为开发的智能助手"
    VERTICAL_SEPERATOR = "|"
    MODULE_VERSION_PREFIX_N2 = "N2"
    TEMPLATE_VERSION_AGENT_V2 = "agent_v2"
    TEMPLATE_VERSION_UNIFY = "unify"

    def __init__(self, llm: LLMApi):
        super(ReactPanguAgent, self).__init__(llm)
        self.set_default_unify_tag()

    def set_default_unify_tag(self):
        """
        给模型提供默认占位符
        """
        llm_module_property = self.llm.get_llm_config().llm_module_config.llm_module_property

        # 优先使用外部主动配置的参数
        if llm_module_property.unify_tool_tag_prefix and llm_module_property.unify_tool_tag_suffix and\
                llm_module_property.unify_tag_prefix and llm_module_property.unify_tag_suffix:
            return

        # 需要整体配置或不配置，不支持仅配置部分参数
        if llm_module_property.unify_tool_tag_prefix or llm_module_property.unify_tool_tag_suffix or\
                llm_module_property.unify_tag_prefix or llm_module_property.unify_tag_suffix:
            raise ValueError("Some unify tags are not configured.")

        # N2占位符为尖括号形式，后续版本为中括号形式
        if self.is_n2_module():
            llm_module_property.unify_tag_prefix = "<unused0>"
            llm_module_property.unify_tag_suffix = "<unused1>"
            llm_module_property.unify_tool_tag_prefix = "<unused2>"
            llm_module_property.unify_tool_tag_suffix = "<unused3>"
        else:
            llm_module_property.unify_tag_prefix = "[unused9]"
            llm_module_property.unify_tag_suffix = "[unused10]"
            llm_module_property.unify_tool_tag_prefix = "[unused11]"
            llm_module_property.unify_tool_tag_suffix = "[unused12]"

    def react(self, agent_session: AgentSession):
        # 超过最大迭代次数限制，不再执行
        if self.need_interrupt(agent_session):
            return

        # 构造React prompt
        react_tp = self.get_react_template()
        messages = agent_session.messages
        default_sys_prompt = self.DEFAULT_SYS_PROMPT if self.is_plugin_v1_version(self.get_template_version()) else ""
        sys_prompt = self.get_system_prompt(
            agent_session) if self.get_system_prompt(agent_session) is not None else default_sys_prompt
        final_prompt = react_tp.format(sys_prompt=sys_prompt,
                                       tool_desc=self.get_tool_desc(),
                                       messages=self.convert_message_to_dict(messages),
                                       cot_desc=self.llm.get_llm_config().llm_module_config.cot_desc)
        normalize_prompt = self.normalize_prompt_template(final_prompt)
        # 调用llm
        if self.llm.get_llm_config().llm_param_config.stream:
            tokens = self.llm.ask(normalize_prompt)
            answer = ""
            for token in tokens:
                answer += token
        else:
            answer = self.llm.ask(normalize_prompt).answer
        llm_module_property = self.llm.get_llm_config().llm_module_config.llm_module_property

        # 获取工具，例如：reserve_meeting_room|{'meetingRoom':'2303','start':'03:00','end':'08:00'}\n\n
        tool_use = self.sub_str_before(self.sub_str_between(answer,
                                                            llm_module_property.unify_tool_tag_prefix + "工具调用:",
                                                            llm_module_property.unify_tool_tag_suffix),
                                       llm_module_property.unify_tool_tag_prefix)
        # 新版本直接使用tool_tag_prefix，没有 工具调用: 的描述
        if not tool_use:
            tool_use = self.sub_str_before(self.sub_str_between(answer,
                                                                llm_module_property.unify_tool_tag_prefix,
                                                                llm_module_property.unify_tool_tag_suffix),
                                           llm_module_property.unify_tool_tag_prefix)
        tool_id = self.sub_str_before(tool_use, "|")
        # 未找到工具则返回
        if tool_id == "":
            action = AgentAction(req=normalize_prompt,
                                 resp=answer,
                                 thought=answer,
                                 action=AgentSessionHelper.FINAL_ACTION,
                                 action_input=answer)
            agent_session.current_action = action
            self.notice_session_end(agent_session, action)
            return
        tool = self.get_tool(tool_id)
        action = AgentAction(req=normalize_prompt,
                             resp=answer,
                             thought=self.sub_str_before(answer, llm_module_property.unify_tool_tag_prefix),
                             action_json="",
                             action=tool_id)
        agent_session.current_action = action

        # 提取工具参数
        action.action_input = self.sub_str_after(tool_use, self.VERTICAL_SEPERATOR).strip(self.VERTICAL_SEPERATOR)
        try:
            if tool.input_type in [int, float, str, bool]:
                json_obj = json.loads(action.action_input)
                if not json_obj or len(json_obj.values()) != 1:
                    raise ValueError(f"the action input is not a single input, require: {tool.get_pangu_function()},"
                                     f" action return: {action.action_input}")
                # 这里添加容错，对单个参数的字段名不做限制{}
                tool_input = list(json_obj.values())[0]
            elif tool.input_type is None:
                tool_input = "{}"
            else:
                tool_input = json.loads(action.action_input)
        except JSONDecodeError:
            tool_input = action.action_input

        # 执行工具
        self.tool_execute(tool, tool_input, agent_session)
        logger.info("actions = %s",
                    "\n".join([action.json(ensure_ascii=False) for action in agent_session.current_message.actions]))

    def get_tool_desc(self):
        return self.get_tool_desc_template().format(tools=[
            {"panguFunction": self.tool_map[tool].get_pangu_function()} for tool in self.tool_map])

    def normalize_prompt_template(self, prompt_str: str) -> str:
        """
        根据实际模型配置的占位符替换prompt模板中的占位符
        :param prompt_str: 输入prompt
        :return: 更换占位符后prompt
        """
        llm_module_property = self.llm.get_llm_config().llm_module_config.llm_module_property
        if llm_module_property and llm_module_property.unify_tag_prefix and\
                llm_module_property.unify_tag_prefix != "[unused9]":
            return prompt_str.replace("[unused9]", llm_module_property.unify_tag_prefix)\
                .replace("[unused10]", llm_module_property.unify_tag_suffix) \
                .replace("[unused11]", llm_module_property.unify_tool_tag_prefix) \
                .replace("[unused12]", llm_module_property.unify_tool_tag_suffix)
        return prompt_str

    def is_n2_module(self) -> bool:
        module_version = self.llm.get_llm_config().llm_module_config.module_version
        return module_version.startswith("38B") or module_version.startswith(self.MODULE_VERSION_PREFIX_N2)

    def is_plugin_v1_version(self, template_version) -> bool:
        return template_version not in [self.TEMPLATE_VERSION_AGENT_V2, self.TEMPLATE_VERSION_UNIFY]

    def get_template_version(self) -> str:
        module_version = self.llm.get_llm_config().llm_module_config.module_version
        return self.sub_str_after(module_version, "_")

    def get_react_template(self) -> PromptTemplate:
        template_version = self.get_template_version()
        if template_version == self.TEMPLATE_VERSION_AGENT_V2:
            return PromptTemplates.get("agent_react_pangu_2")
        elif template_version == self.TEMPLATE_VERSION_UNIFY:
            return PromptTemplates.get("agent_react_pangu_unify")
        else:
            return PromptTemplates.get("agent_react_pangu")

    def get_tool_desc_template(self) -> PromptTemplate:
        template_version = self.get_template_version()
        if template_version == self.TEMPLATE_VERSION_AGENT_V2 or template_version == self.TEMPLATE_VERSION_UNIFY:
            return PromptTemplates.get("agent_tool_desc_pangu_2")
        else:
            return PromptTemplates.get("agent_tool_desc_pangu")

    def set_cot_desc(self, cot_desc: str):
        self.llm.get_llm_config().llm_module_config.cot_desc = cot_desc
