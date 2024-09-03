#  Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
from pangukitsappdev.api.llms.base import LLMApi
from pangukitsappdev.api.skill.base import SimpleSkill
from pangukitsappdev.prompt.prompt_tmpl import PromptTemplates


class DocAskSkill(SimpleSkill):
    def __init__(self, llm_api: LLMApi):
        super().__init__(PromptTemplates.get("documents_stuff"), llm_api)
