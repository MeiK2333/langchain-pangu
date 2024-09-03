#  Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
from abc import ABC, abstractmethod


class PythonSandBox(ABC):
    @abstractmethod
    def run(self, code: str) -> str:
        """
        执行一段Python代码
        :param code: Python代码
        :return: Python代码的执行结果
        """