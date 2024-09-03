#  Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
from abc import ABC, abstractmethod
from typing import List

from langchain_pangu.pangukitsappdev.api.memory.vector.base import Document


class KnowledgeGraph(ABC):
    @abstractmethod
    def query(self, query: str) -> List[Document]:
        """
        查找知识图谱
        :param query: 查询体
        :return: 相关回答
        """