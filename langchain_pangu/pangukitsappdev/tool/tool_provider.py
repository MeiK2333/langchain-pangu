#  Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
from abc import abstractmethod, ABC
from typing import List

from langchain_pangu.pangukitsappdev.api.tool.base import AbstractTool
from langchain_pangu.pangukitsappdev.retriever.retrieved_tool import RetrievedTool


class ToolProvider(ABC):
    """工具实例化提供者，通过toolId列表给出实例化工具
    """
    @abstractmethod
    def provide(self, retrieved_tools: List[RetrievedTool], query: str) -> List[AbstractTool]:
        """通过id查找工具
        Args:
            retrieved_tools: 通过ToolRetriever召回的工具
            query: 查询语句
        Returns: 实例化后的工具
        """
