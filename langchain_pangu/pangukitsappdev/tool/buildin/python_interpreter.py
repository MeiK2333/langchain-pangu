#  Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
from typing import Optional, Type

from pydantic.v1 import Field, BaseModel

from langchain_pangu.pangukitsappdev.tool.python_sandbox import PythonSandBox
from langchain_pangu.pangukitsappdev.tool.tool import Tool


class PythonInterpreterTool(Tool):
    class InputParam(BaseModel):
        code: str = Field(description="Python代码")

    name = "python_interpreter"
    description = "通过调用python解释器执行代码来解决计算和文件操作类问题"
    principle = "请在需要进行数学计算、日期查询或者文件操作等场景使用此工具"
    args_schema: Type[BaseModel] = InputParam
    output_desc = "代码执行结果"
    python_sandbox: Optional[PythonSandBox]

    def _run(self, code: str) -> str:
        if self.python_sandbox is None:
            raise ValueError("Param python_sandbox must be specified")
        return self.python_sandbox.run(code)
