#  Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
from typing import Optional

from langchain_pangu.pangukitsappdev.tool.python_sandbox import PythonSandBox
from langchain_pangu.pangukitsappdev.tool.tool import Tool


class UnifyPythonInterpreterTool(Tool):

    name = "python_interpreter"
    description = ""
    input_desc = ""
    output_desc = ""
    principle = ""
    python_sandbox: Optional[PythonSandBox]
    pangu_function = '{"name": "python_interpreter", "description:": "python解释器",' \
                     ' "principle:": "问题可用编程解决时使用",' \
                     ' "arguments": "String: python代码", "results": "String: 执行结果"}'

    def _run(self, code: str) -> str:
        if self.python_sandbox is None:
            raise ValueError("Param python_sandbox must be specified")
        return self.python_sandbox.run(code)
