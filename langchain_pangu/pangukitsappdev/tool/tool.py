#  Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
from typing import Optional, Type, Union, Callable, Any, get_origin, get_args

from pydantic.v1 import BaseModel, create_model, Field
from pydantic.v1.tools import schema_of
from pydantic.v1.typing import NoneType

from langchain_pangu.pangukitsappdev.api.tool.base import AbstractTool, PanguFunction, DEFAULT_SINGLE_ARG
from langchain_pangu.pangukitsappdev.prompt.prompt_tmpl import PromptTemplates


class Tool(AbstractTool):
    tool_input_schema: Optional[str]
    tool_output_schema: Optional[str]
    pangu_function: Optional[str]

    def _run(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Use the tool."""
        if self.func:
            return self.func(*args, **kwargs)
        raise NotImplementedError("Tool does not support sync")

    def get_input_schema(self) -> str:
        if not self.tool_input_schema:
            self.tool_input_schema = self.build_tool_schema(self.input_desc, self.input_type)
        return self.tool_input_schema

    def get_output_schema(self) -> str:
        if not self.tool_output_schema:
            self.tool_output_schema = self.build_tool_schema(self.output_desc, self.output_type)
        return self.tool_output_schema

    def get_pangu_function(self) -> str:
        if not self.pangu_function:
            self.pangu_function = self.build_pangu_function()
        return self.pangu_function

    @staticmethod
    def build_tool_schema(desc: str, class_type: Union[Type, BaseModel]):
        desc = "空" if class_type == NoneType else desc
        if class_type in [int, float, str, bool, list, dict, NoneType]:
            return PromptTemplates.get("agent_tool_simple_schema").format(desc=desc,
                                                                          type=class_type.__name__)
        # 复杂类型返回JSON schema
        return PromptTemplates.get("agent_tool_json_schema"). \
            format(desc=desc, schema={"properties": class_type.schema().get('properties')})

    def build_pangu_function(self):
        # 诺亚新版本模型并未对principle / results进行训练，临时拼接到description后面
        # principle / results不填值
        arguments = self.get_pang_tool_schema(self.input_desc, self.input_type)
        results = self.get_pang_tool_schema(self.output_desc, self.output_type)
        description = (self.get_tool_desc() + "，" + self.principle) if self.principle else self.get_tool_desc()
        return PanguFunction(name=self.get_tool_id(),
                             description=description,
                             arguments=arguments,
                             results=results).json(ensure_ascii=False, exclude_none=True)

    def get_pang_tool_schema(self, desc: str, class_type: Union[Type, BaseModel]) -> Any:
        # 基本类型
        if class_type in [int, float, str, bool]:
            result_schema = create_model(DEFAULT_SINGLE_ARG, arg=(class_type, Field(description=desc))).schema()
        elif get_origin(class_type) in [list, dict]:
            result_schema = schema_of(class_type)
            result_schema.pop("items", None)
            if len(get_args(class_type)) < 1:
                raise ValueError("List or Dict must specify item type")
            item_type = get_args(class_type)[-1]
            if item_type in [int, float, str, bool, NoneType]:
                result_schema.update({"items": {"type": item_type.__name__}})
            else:
                result_schema["items"] = self.get_pang_tool_schema(desc=desc, class_type=item_type)
        elif class_type is NoneType:
            result_schema = schema_of(class_type)
        else:
            # 复杂类型(BaseModel子类)
            result_schema = class_type.schema()

        definitions = result_schema.pop("definitions", {})
        # 移除冗余title
        result_schema.pop('title', None)
        for prop_key in result_schema.get('properties', {}):
            prop = result_schema.get('properties', {}).get(prop_key, {})
            prop.pop('title', None)
            all_of = prop.pop("allOf", [])
            if all_of:
                enum_name = all_of[0].get('$ref', "").split('/')[-1]
                if definitions.get(enum_name, {}).get("enum") and definitions.get(enum_name, {}).get("type"):
                    prop["enum"] = definitions.get(enum_name, {}).get("enum")
                    prop["type"] = definitions.get(enum_name, {}).get("type")
        return result_schema

    @classmethod
    def from_function(
            cls,
            func: Optional[Callable],
            name: str,
            description: str,
            principle: str,
            input_desc: str,
            output_desc: str,
            args_schema: Optional[Type[BaseModel]] = None,
            return_type: Optional[Type] = None
    ) -> AbstractTool:
        """Initialize tool from a function."""
        if func is None:
            raise ValueError("Function must be provided")
        return cls(
            name=name,
            func=func,
            description=description,
            principle=principle,
            input_desc=input_desc,
            output_desc=output_desc,
            args_schema=args_schema,
            return_type=return_type
        )
