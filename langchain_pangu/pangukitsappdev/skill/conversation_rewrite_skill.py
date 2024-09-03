#  Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
from typing import List
from pangukitsappdev.api.skill.base import SimpleSkill
from pangukitsappdev.api.llms.base import LLMApi, ConversationMessage
from pangukitsappdev.prompt.prompt_tmpl import PromptTemplates
from langchain.prompts import PromptTemplate
import logging

logger = logging.getLogger(__name__)


class ConversationRewriteSkill:

    def __init__(self, llm: LLMApi):
        self.llm = llm
        self.prompt = PromptTemplates.get("skill_conversation_rewrite")

    def set_prompt(self, prompt_template: str):
        self.prompt = PromptTemplate.from_template(prompt_template, template_format="jinja2")

    def rewrite(self, messages: List[ConversationMessage]) -> str:
        query_write = SimpleSkill(self.prompt, self.llm)
        rewrite_result = query_write.execute({"messages": messages})
        logger.info("ConversationRewrite result: %s", rewrite_result)
        return rewrite_result
