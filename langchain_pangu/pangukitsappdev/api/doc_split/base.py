#  Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.

from abc import ABC, abstractmethod
from typing import List, Iterator

from langchain.document_loaders.base import BaseLoader
from langchain.schema import Document

from langchain_pangu.pangukitsappdev.api.doc_split.split_config import SplitConfig


class LoaderApi(BaseLoader, ABC):
    @abstractmethod
    def load(self) -> List[Document]:
        """
        对文档进行加载并解析分割
        :return: 分割的document列表
        """
        pass


class AbstractLoaderApi(LoaderApi, ABC):

    def __init__(self, split_config: SplitConfig):
        self.split_config = split_config

    @abstractmethod
    def load(self) -> List[Document]:
        """
        实现类方法
        :return: 解析分割后的document列表
        """
        pass


    def lazy_load(self) -> Iterator[Document]:
        # 默认未实现，直接抛异常
        raise NotImplementedError("Unimplement method!")