#  Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
from typing import List
from pangukitsappdev.api.memory.vector.vector_config import VectorStoreConfig
from pangukitsappdev.api.memory.vector.factory import Vectors
from pangukitsappdev.api.memory.vector.base import VectorApi
from pangukitsappdev.api.retriever.base import AbstractToolRetriever
from pangukitsappdev.api.tool.base import AbstractTool
from pangukitsappdev.tool.tool_provider import ToolProvider
from pangukitsappdev.retriever.tool_metadata import ToolMetadata


class ToolRetrieverImpl(AbstractToolRetriever):

    def __init__(self, tool_provider: ToolProvider, vector_store_config: VectorStoreConfig):
        """储存工具类相关信息
        初始化
        :param tool_provider: 工具存储
        :param: vector_store_config: 语义向量存储配置
        """
        super().__init__(tool_provider, vector_store_config)
        self.vector: VectorApi = Vectors.of(vector_config=vector_store_config)

    def add_tools(self, tools: List[AbstractTool]):
        if not self.vector_store_config.vector_fields:
            raise ValueError("vector_store_config must specify vector_fields!")
        tool_metadata_list = [ToolMetadata(tool_id=tool.get_tool_id(),
                                           tool_metadata={
                                               self.vector_store_config.vector_fields[0]:
                                                   tool.get_tool_desc() + "，" + tool.principle})
                              for tool in tools]
        self.add_tools_from_metadata(tool_metadata_list)

    def dsl_search(self, query: str, dsl: str) -> List[AbstractTool]:
        raise NotImplementedError("Unsupported operation!")