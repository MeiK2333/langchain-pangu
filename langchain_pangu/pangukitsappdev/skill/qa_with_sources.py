#  Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
from langchain_pangu.pangukitsappdev.api.llms.base import LLMApi
from langchain_pangu.pangukitsappdev.api.skill.base import SimpleSkill
from langchain_pangu.pangukitsappdev.prompt.prompt_tmpl import PromptTemplates


class DocAskSkill(SimpleSkill):
    def __init__(self, llm_api: LLMApi):
        super().__init__(PromptTemplates.get("documents_stuff"), llm_api)
