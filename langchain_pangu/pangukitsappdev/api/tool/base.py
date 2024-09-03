#  Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
from abc import ABC, abstractmethod
from inspect import signature, Signature
from typing import Optional, Any, Callable, Type

from langchain.tools import BaseTool
from pydantic.v1 import BaseModel

DEFAULT_SINGLE_ARG = "arg"


class AbstractTool(BaseTool, ABC):
    input_desc: str
    output_desc: str
    principle: str
    func: Optional[Callable[..., str]]
    return_type: Optional[Type]

    def get_tool_id(self):
        """
        工具的唯一标识，在同一个Agent里必须唯一
        Returns: str, 工具的标识
        """
        return self.name

    def get_tool_desc(self) -> str:
        """
        工具的描述，唯一值
        Returns: str, 工具的描述
        """
        return self.description

    @property
    def input_type(self):
        """
        工具的input类型，
        Returns: Type, 工具input类型
        """
        if self.args_schema is not None:
            return self.args_schema
        else:
            function = self.func if self.func else self._run
            params = list(signature(function).parameters.values())
            if len(params) != 1 and len(params) != 0:
                raise ValueError("Input param should extend BaseTool or have less than one simple type")
            return params[0].annotation if params else type(None)

    @property
    def output_type(self):
        """
        工具的返回值类型，此方法可获取func或_run方法返回类型
        Returns: Type, 工具的返回值类型
        """
        if self.func:
            # 动态工具获取self.func类型
            func_type = signature(self.func).return_annotation
        else:
            # 静态工具重写self._run方法指定类型
            func_type = signature(self._run).return_annotation
        # 若为lambda表达式或未指定返回类型函数，需要创建时指定return_type
        self.return_type = self.return_type if func_type == Signature.empty else func_type
        if not self.return_type:
            raise ValueError("Dynamic func method or static _run method must specify return annotation!")
        return self.return_type

    @abstractmethod
    def get_input_schema(self) -> str:
        """
        获取入参的Json Schema定义
        Returns: str, 获取入参的Json Schema定义
        """

    @abstractmethod
    def get_output_schema(self) -> str:
        """
        获取出参的Json Schema定义
        Returns: str, 获取出参的Json Schema定义
        """

    @abstractmethod
    def get_pangu_function(self) -> str:
        """
        获取pangu Function的定义
        Returns: str, pangu Function的定义
        """


class PanguFunction(BaseModel):
    """PanguFunction参数
    Attributes:
        name: tool名称
        description: tool功能描述，描述tool的作用
        arguments: tool输入
        principle: tool使用原则，告诉模型在什么情况下使用tool
        results: tool输出
    """
    name: Optional[str]
    description: Optional[str]
    arguments: Optional[Any]
    principle: Optional[str]
    results: Optional[Any]
