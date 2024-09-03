#  Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
from typing import List

from langchain_pangu.pangukitsappdev.api.tool.base import AbstractTool
from langchain_pangu.pangukitsappdev.retriever.retrieved_tool import RetrievedTool
from langchain_pangu.pangukitsappdev.tool.tool_provider import ToolProvider


class InMemoryToolProvider(ToolProvider):
    """工具类持久化(内存)
    """
    def __init__(self):
        self.tool_store: dict[str, AbstractTool] = {}

    def provide(self, retrieved_tools: List[RetrievedTool], query: str) -> List[AbstractTool]:
        tools = []
        for retrieved_tool in retrieved_tools:
            if retrieved_tool.tool_id in self.tool_store.keys():
                tools.append(self.tool_store.get(retrieved_tool.tool_id))
        return tools

    def add(self, tools: List[AbstractTool]) -> List[str]:
        tool_ids = []
        for tool in tools:
            tool_ids.append(tool.get_tool_id())
            self.tool_store.update({tool.get_tool_id(): tool})
        return tool_ids

    def remove(self, tool_ids: List[str]):
        for tool_id in tool_ids:
            self.tool_store.pop(tool_id)
