#  Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
from typing import Dict, Type

from langchain_pangu.pangukitsappdev.api.doc_split.base import LoaderApi
from langchain_pangu.pangukitsappdev.api.doc_split.split_config import SplitConfig
from langchain_pangu.pangukitsappdev.doc_split.pangu_split import DocPanguSplit


class DocSplits:
    split_map: Dict[str, Type[LoaderApi]] = {}

    @classmethod
    def register(cls, split_type: Type[LoaderApi], split_name: str):
        """
        注册一种split的类型
        :param split_type: split的类型，要求是LoaderApi的子类
        :param split_name: split的名字，唯一代表这个split的名字
        :return: none
        """
        cls.split_map[split_name] = split_type

    @classmethod
    def of(cls, split_name: str, split_config: SplitConfig = None) -> LoaderApi:
        """
        根据名字创建一个LoaderApi的实现类
        :param split_name: split的名字，唯一标识一种分割
        :param split_config: （Optional）split的相关配置，如果不传递则从默认配置文件中或者环境变量中获取
        :return: LoaderApi
        """

        split_type = cls.split_map.get(split_name)
        if not split_type:
            raise ValueError(
                f"Unregistered split name: {split_name}, "
                f"please call register(split_type, split_name) before use.")

        local_split_config = split_config if split_config else cls._load_split_config()

        return split_type(local_split_config)

    @classmethod
    def _load_split_config(cls):
        return SplitConfig()


DocSplits.register(DocPanguSplit, "pangu-doc")
