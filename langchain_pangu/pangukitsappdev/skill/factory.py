#  Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
from typing import Dict

from langchain.prompts import PromptTemplate

from langchain_pangu.pangukitsappdev.api.llms.base import LLMApi
from langchain_pangu.pangukitsappdev.api.llms.factory import LLMs
from langchain_pangu.pangukitsappdev.api.skill.base import SimpleSkill
from langchain_pangu.pangukitsappdev.skill.qa_with_sources import DocAskSkill


class DocumentSkills:
    """
    构造文档相关的Executor
    """

    prompt_template_map: Dict[str, PromptTemplate] = {}

    @classmethod
    def register_prompt_template(cls, skill_type, prompt_template: PromptTemplate):
        cls.prompt_template_map[skill_type] = prompt_template

    @classmethod
    def of(cls, llm_api: LLMApi) -> DocAskSkill:
        return DocAskSkill(llm_api)

    @classmethod
    def of_type(cls, skill_type: str, llm_api: LLMApi) -> DocAskSkill:
        pt = cls.prompt_template_map.get(skill_type)
        if not pt:
            raise ValueError(
                f"Unregistered skill type: {skill_type}, "
                f"please call register_prompt_template(skill_type, prompt_template) before use.")

        local_llm_api = llm_api if llm_api else LLMs.of("pangu")

        return SimpleSkill(local_llm_api, pt)
