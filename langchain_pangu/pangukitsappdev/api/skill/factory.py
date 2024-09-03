#  Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
from langchain.chains.llm import LLMChain
from langchain.schema.prompt_template import BasePromptTemplate

from langchain_pangu.pangukitsappdev.api.llms.base import LLMApi
from langchain_pangu.pangukitsappdev.api.skill.base import ChainWrappedSkill, SimpleSkill


class Skills:
    """
    静态工厂类，用来创建Skill
    """
    @classmethod
    def of_chain(cls, chain: LLMChain) -> ChainWrappedSkill:
        return ChainWrappedSkill(chain)

    @classmethod
    def of(cls, prompt_template: BasePromptTemplate, llm_api: LLMApi):
        return SimpleSkill(prompt_template, llm_api)
