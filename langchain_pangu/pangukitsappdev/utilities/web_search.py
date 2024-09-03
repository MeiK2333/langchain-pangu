#  Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
from abc import ABC, abstractmethod
from typing import List

from langchain_pangu.pangukitsappdev.api.memory.vector.base import Document


class WebSearch(ABC):
    @abstractmethod
    def query(self, query: str, top_k: int) -> List[Document]:
        """
        执行query
        :param query: 查询体
        :param top_k: 查询记录数
        :return: 相关文档
        """