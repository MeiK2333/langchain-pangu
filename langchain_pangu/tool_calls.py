#  Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
"""
此文件大部分内容是从盘古 sdk react_pangu_agent.py 复制而来，因为盘古 sdk 将 tools 操作封装成 agent，无法灵活对接 langchain，因此抽出来重新实现
"""
import json
from typing import List, Union, Dict, Any, Type, Callable

from langchain_core.messages import BaseMessage, ToolCall
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import BaseModel

from langchain_pangu.pangukitsappdev.api.llms.llm_config import LLMConfig
from langchain_pangu.pangukitsappdev.api.tool.base import PanguFunction
from langchain_pangu.pangukitsappdev.prompt.prompt_tmpl import PromptTemplates


class PanguToolCalls:
    DEFAULT_SYS_PROMPT = "你的名字叫智子，是由华为开发的智能助手"
    VERTICAL_SEPERATOR = "|"
    MODULE_VERSION_PREFIX_N2 = "N2"
    TEMPLATE_VERSION_AGENT_V2 = "agent_v2"
    TEMPLATE_VERSION_UNIFY = "unify"

    def __init__(self, llm_config: LLMConfig):
        self.llm_config = llm_config
        self.tools: List[Union[Dict[str, Any], Type[BaseModel], Callable, BaseTool]] = (
            []
        )
        self.set_default_unify_tag()

    def set_default_unify_tag(self):
        """
        给模型提供默认占位符
        """
        llm_module_property = self.llm_config.llm_module_config.llm_module_property

        # 优先使用外部主动配置的参数
        if (
            llm_module_property.unify_tool_tag_prefix
            and llm_module_property.unify_tool_tag_suffix
            and llm_module_property.unify_tag_prefix
            and llm_module_property.unify_tag_suffix
        ):
            return

        # 需要整体配置或不配置，不支持仅配置部分参数
        if (
            llm_module_property.unify_tool_tag_prefix
            or llm_module_property.unify_tool_tag_suffix
            or llm_module_property.unify_tag_prefix
            or llm_module_property.unify_tag_suffix
        ):
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

    def is_n2_module(self) -> bool:
        module_version = self.llm_config.llm_module_config.module_version
        return module_version.startswith("38B") or module_version.startswith(
            self.MODULE_VERSION_PREFIX_N2
        )

    def add_tool(
        self, tool: Union[Dict[str, Any], Type[BaseModel], Callable, BaseTool]
    ):
        self.tools.append(tool)

    def remove_tool(
        self, tool: Union[Dict[str, Any], Type[BaseModel], Callable, BaseTool]
    ):
        self.tools.remove(tool)

    def clear_tool(self):
        self.tools.clear()

    def tool_calls_prompt(self, messages: List[BaseMessage]):
        react_tp = self.get_react_template()

        default_sys_prompt = (
            self.DEFAULT_SYS_PROMPT
            if self.is_plugin_v1_version(self.get_template_version())
            else ""
        )
        sys_prompt = self.get_system_prompt(messages)
        if sys_prompt is None:
            sys_prompt = default_sys_prompt
        final_prompt = react_tp.format(
            sys_prompt=sys_prompt,
            tool_desc=self.get_tool_desc(),
            messages=self.convert_message_to_dict(messages),
            cot_desc=self.llm_config.llm_module_config.cot_desc,
        )
        normalize_prompt = self.normalize_prompt_template(final_prompt)
        return normalize_prompt

    def tool_calls(self, text: str) -> List[ToolCall]:
        llm_module_property = self.llm_config.llm_module_config.llm_module_property

        # 获取工具，例如：reserve_meeting_room|{'meetingRoom':'2303','start':'03:00','end':'08:00'}\n\n
        tool_use = self.sub_str_before(
            self.sub_str_between(
                text,
                llm_module_property.unify_tool_tag_prefix + "工具调用:",
                llm_module_property.unify_tool_tag_suffix,
            ),
            llm_module_property.unify_tool_tag_prefix,
        )
        # 新版本直接使用tool_tag_prefix，没有 工具调用: 的描述
        if not tool_use:
            tool_use = self.sub_str_before(
                self.sub_str_between(
                    text,
                    llm_module_property.unify_tool_tag_prefix,
                    llm_module_property.unify_tool_tag_suffix,
                ),
                llm_module_property.unify_tool_tag_prefix,
            )

        if not tool_use:
            return []
        if "|" not in tool_use:
            return []
        tool_name, tool_params = tool_use.split("|", 2)
        # 判断对应 tool 是否存在
        for t in self.tools:
            if t.name == tool_name:
                break
        else:
            return []
        return [
            ToolCall(
                name=tool_name,
                id=tool_name,
                args=json.loads(tool_params),
                type="tool_call",
            )
        ]

    @staticmethod
    def sub_str_between(origin_str: str, start_str: str, end_str: str):
        if origin_str:
            start_pos = origin_str.find(start_str)
            if start_pos != -1:
                end_pos = origin_str.find(end_str)
                if end_pos != -1:
                    return origin_str[start_pos + len(start_str) : end_pos]
        return ""

    @staticmethod
    def sub_str_before(origin_str: str, separator: str):
        if origin_str:
            if not separator:
                return ""
            else:
                pos = origin_str.find(separator)
                return origin_str if pos == -1 else origin_str[:pos]
        else:
            return origin_str

    @staticmethod
    def convert_message_to_dict(conversation_messages: List[BaseMessage]) -> List[Dict]:
        # 用于将python命名风格变量适配小驼峰，与Java版本保持prompt模板统一
        return [
            {
                "role": {"text": message.type, "desc": message.type},
                "content": message.content,
            }
            for message in conversation_messages
        ]

    def normalize_prompt_template(self, prompt_str: str) -> str:
        """
        根据实际模型配置的占位符替换prompt模板中的占位符
        :param prompt_str: 输入prompt
        :return: 更换占位符后prompt
        """
        llm_module_property = self.llm_config.llm_module_config.llm_module_property
        if (
            llm_module_property
            and llm_module_property.unify_tag_prefix
            and llm_module_property.unify_tag_prefix != "[unused9]"
        ):
            return (
                prompt_str.replace("[unused9]", llm_module_property.unify_tag_prefix)
                .replace("[unused10]", llm_module_property.unify_tag_suffix)
                .replace("[unused11]", llm_module_property.unify_tool_tag_prefix)
                .replace("[unused12]", llm_module_property.unify_tool_tag_suffix)
            )
        return prompt_str

    def get_system_prompt(self, messages: List[BaseMessage]):
        # 优先取设置到的LLM人设
        system_prompt = self.llm_config.llm_module_config.system_prompt
        if system_prompt is not None:
            return system_prompt
        # 其次取message中最后一个人设
        for message in reversed(messages):
            if message.type == "system":
                return message.content
        return None

    def is_plugin_v1_version(self, template_version) -> bool:
        return template_version not in [
            self.TEMPLATE_VERSION_AGENT_V2,
            self.TEMPLATE_VERSION_UNIFY,
        ]

    def get_react_template(self) -> PromptTemplate:
        template_version = self.get_template_version()
        if template_version == self.TEMPLATE_VERSION_AGENT_V2:
            return PromptTemplates.get("agent_react_pangu_2")
        elif template_version == self.TEMPLATE_VERSION_UNIFY:
            return PromptTemplates.get("agent_react_pangu_unify")
        else:
            return PromptTemplates.get("agent_react_pangu")

    @staticmethod
    def sub_str_after(origin_str: str, separator: str):
        if origin_str:
            if separator == "":
                return ""
            else:
                pos = origin_str.find(separator)
                return "" if pos == -1 else origin_str[pos + len(separator) :]
        else:
            return origin_str

    def get_template_version(self) -> str:
        module_version = self.llm_config.llm_module_config.module_version
        return self.sub_str_after(module_version, "_")

    def get_tool_desc_template(self) -> PromptTemplate:
        template_version = self.get_template_version()
        if (
            template_version == self.TEMPLATE_VERSION_AGENT_V2
            or template_version == self.TEMPLATE_VERSION_UNIFY
        ):
            return PromptTemplates.get("agent_tool_desc_pangu_2")
        else:
            return PromptTemplates.get("agent_tool_desc_pangu")

    def get_tool_desc(self):
        tools = []
        for tool in self.tools:
            formatted_tool = convert_to_openai_tool(tool)
            if "function" not in formatted_tool:
                continue
            formatted_func = formatted_tool["function"]
            func = PanguFunction(
                name=formatted_func["name"],
                description=formatted_func["description"],
                arguments=formatted_func["parameters"],
                results={},
            ).json(ensure_ascii=False, exclude_none=True)
            tools.append({"panguFunction": func})
        return self.get_tool_desc_template().format(tools=tools)
