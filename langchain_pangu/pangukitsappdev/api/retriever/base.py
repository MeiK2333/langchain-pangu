#  Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
from abc import ABC, abstractmethod
from typing import List, Optional, Callable

from langchain_pangu.pangukitsappdev.api.llms.base import ConversationMessage
from langchain_pangu.pangukitsappdev.api.memory.vector.base import VectorApi, Document
from langchain_pangu.pangukitsappdev.api.memory.vector.vector_config import VectorStoreConfig
from langchain_pangu.pangukitsappdev.api.tool.base import AbstractTool
from langchain_pangu.pangukitsappdev.retriever.retrieved_tool import RetrievedTool
from langchain_pangu.pangukitsappdev.retriever.tool_metadata import ToolMetadata
from langchain_pangu.pangukitsappdev.tool.tool_provider import ToolProvider
from langchain_pangu.pangukitsappdev.vectorstores.bulk_data import BulkData


class ToolRetriever(ABC):
    """工具检索器接口类
    """

    @abstractmethod
    def add_tools(self, tools: List[AbstractTool]):
        """
        工具入库
        :param tools: tools工具
        """

    @abstractmethod
    def add_tools_from_metadata(self, tool_metadata_list: List[ToolMetadata]):
        """
        工具入库
        :param tool_metadata_list: 需要入库的工具
        """

    @abstractmethod
    def search(self, query: str,
               top_k: Optional[int] = None,
               score_threshold: Optional[float] = None) -> List[AbstractTool]:
        """
        工具检索
        :param query: 查询语句
        :param top_k: top k条
        :param score_threshold: 评分阈值
        :return: 相似工具列表
        """

    @abstractmethod
    def remove(self, tool_ids: List[str]):
        """
        删除工具
        :param tool_ids: tools工具
        """

    @abstractmethod
    def set_query_preprocessor(self, preprocessor: Callable[[List[ConversationMessage]], str]):
        """
        设置query请求的预处理器
        :param preprocessor: 当ToolRetriever被设置在Agent中时，Agent在Retrieve工具前，
        先会调用query预处理器对多轮对话进行改写，如果不设置，默认使用多轮中的最后一轮对话
        """

    @abstractmethod
    def get_query_preprocessor(self) -> Callable[[List[ConversationMessage]], str]:
        """
        获取query请求的预处理器
        :return: query预处理器
        """

    @abstractmethod
    def dsl_search(self, query: str, dsl: str) -> List[AbstractTool]:
        """
        工具检索
        :param query : 查询语句
        :param dsl : dsl
        :return: 相似工具列表
        """


class AbstractToolRetriever(ToolRetriever, ABC):
    RETRIEVER_SIZE_DEFAULT = 5
    RETRIEVER_SCORE_DEFAULT = 0.2

    def __init__(self, tool_provider: ToolProvider, vector_store_config: VectorStoreConfig):
        """储存工具类相关信息
        初始化
        :param tool_provider: 工具存储
        :param: vector_store_config: 语义向量存储配置
        """
        self.tool_provider: ToolProvider = tool_provider
        self.vector_store_config: VectorStoreConfig = vector_store_config
        self.vector: Optional[VectorApi] = None

        def default_processor(messages: List[ConversationMessage]) -> str:
            query = ""
            for message in reversed(messages):
                if message.role.text == "user":
                    query = message.content
                    break
            return query

        self.preprocessor: Callable[[List[ConversationMessage]], str] = default_processor

    def set_query_preprocessor(self, preprocessor: Callable[[List[ConversationMessage]], str]):
        self.preprocessor = preprocessor

    def get_query_preprocessor(self) -> Callable[[List[ConversationMessage]], str]:
        return self.preprocessor

    def add_tools_from_metadata(self, tool_metadata_list: List[ToolMetadata]):
        bulk_data_list = [BulkData(id=tool_metadata.tool_id,
                                   data=tool_metadata.tool_metadata) for tool_metadata in tool_metadata_list]
        self.vector.add_docs(bulk_data_list)

    def search(self, query: str,
               top_k: int = RETRIEVER_SIZE_DEFAULT,
               score_threshold: float = RETRIEVER_SCORE_DEFAULT) -> List[AbstractTool]:
        # 检索相关工具，获取id
        docs = self.vector.similarity_search(query=query,
                                             top_k=top_k,
                                             score_threshold=score_threshold)
        return self.instantiation_tool(docs, query)

    def remove(self, tool_ids: List[str]):
        self.vector.remove(tool_ids)

    def instantiation_tool(self, documents: List[Document], query: str) -> List[AbstractTool]:
        retrieved_tools = [RetrievedTool(score=doc.score,
                                         tool_id=doc.id,
                                         tool_metadata=doc.metadata) for doc in documents]
        return self.tool_provider.provide(retrieved_tools, query)
