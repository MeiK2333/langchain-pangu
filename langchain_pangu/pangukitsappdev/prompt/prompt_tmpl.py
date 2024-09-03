#  Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
import json
import logging
import os

from langchain.prompts import PromptTemplate

from langchain_pangu.pangukitsappdev.api.prompt.prompt_config import PromptConfig, PromptTemplatesFileConfig

logger = logging.getLogger(__name__)


def read_prompt_from(tmpl_file) -> PromptTemplate:
    with open(tmpl_file, mode="r", encoding="utf8") as ff:
        txt = ff.read()
    return PromptTemplate.from_template(txt, template_format="jinja2")


class PromptTemplates:
    """获取Prompt模板工厂类
        提供get方法根据名字获取prompt模板。提供register方法从文件中注册一个模板
    Attributes:
        prompt_tmpl_lookup: 类成员变量，维护prompt模板
    """

    prompt_tmpl_lookup: dict = {}

    @classmethod
    def get(cls, tmpl_name) -> PromptTemplate:
        """
        根据tmpl_name获取一个模板，优先自定义模板
        Args:
            tmpl_name: 唯一标识一个prompt模板

        Returns:
            prompt模板PromptTemplate

        """
        custom_config = PromptConfig()
        custom_prompt_path = custom_config.custom_prompt_path
        if prompt_files.get(tmpl_name) is None:
            return cls.prompt_tmpl_lookup.get(tmpl_name)
        if custom_prompt_path and os.path.exists(os.path.join(custom_prompt_path, prompt_files.get(tmpl_name))):
            return read_prompt_from(os.path.join(custom_prompt_path, prompt_files.get(tmpl_name)))
        else:
            return cls.prompt_tmpl_lookup.get(tmpl_name)

    @classmethod
    def register(cls, tmpl_file, tmpl_name=None):
        """从文件内容注册一个prompt模板
        文件的后缀代表这个模板的格式化类型，支持f-string和jinja2。比如有一个模板文件是foo.f-string，则会使用f"xx{}xxx"的方式做格式化
        Args:
            tmpl_file: 指向模板文件的路径。文件的后缀代表模板的格式化类型，支持f-string和jinja2
            tmpl_name: （Optional） 唯一指定一个模板的名称，如果不传则使用不包含后缀的文件名
        """
        local_tmpl_name = tmpl_name if tmpl_name else os.path.basename(tmpl_file).split(".")[0]

        cls.prompt_tmpl_lookup[local_tmpl_name] = read_prompt_from(tmpl_file)

    @classmethod
    def template_names(cls) -> list:
        """
        获取所有可以通过PromptTemplateProvider获取的prompt模板名称
        Returns: list, 所有模板名称
        """
        return list(cls.prompt_tmpl_lookup.keys())


# load prompt settings
prompt_config = PromptConfig()
prompt_file_config = PromptTemplatesFileConfig()

# load default prompt path
py_file_path = os.path.split(__file__)[0]
default_prompt_path = os.path.join(py_file_path, prompt_config.default_prompt_path)

# load prompt name for path dict by BaseSettings.json method
prompt_files = json.loads(prompt_file_config.json())

for key in prompt_files.keys():
    # register related prompts
    prompt_path = os.path.join(default_prompt_path, prompt_files[key])
    PromptTemplates.register(prompt_path, key)
logger.debug("Total load default prompts num: %s", len(prompt_files.keys()))
