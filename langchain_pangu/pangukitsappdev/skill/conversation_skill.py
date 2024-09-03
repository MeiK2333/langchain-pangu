#  Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
from typing import Dict, Any
from pangukitsappdev.api.llms.base import LLMApi
from pangukitsappdev.api.skill.base import Skill
from pangukitsappdev.prompt.prompt_tmpl import PromptTemplates
from langchain.memory.buffer import ConversationBufferMemory
from langchain.schema.memory import BaseMemory
from langchain.prompts import PromptTemplate


class ConversationSkill(Skill):

    class PromptParam:
        MEMORY_HISTORY = "history"
        INPUT = "input"
        OUTPUT = "output"

    def execute_dict(self, inputs: Dict[str, Any]) -> str:
        if self.PromptParam.MEMORY_HISTORY in inputs:
            del inputs[self.PromptParam.MEMORY_HISTORY]
        inputs.update(self.memory.load_memory_variables({}))
        answer = self.skill_llm_ask(self.prompt.format(**inputs), self.llm)
        self.memory.save_context({self.PromptParam.INPUT: inputs.get(self.PromptParam.INPUT)},
                                 {self.PromptParam.OUTPUT: answer})
        return answer

    def __init__(self, llm: LLMApi):
        self.llm = llm
        self.memory = ConversationBufferMemory()
        self.prompt = PromptTemplates.get("conversation_default")

    def set_prompt(self, prompt_template: str):
        self.prompt = PromptTemplate.from_template(prompt_template, template_format="jinja2")

    def set_memory(self, memory: BaseMemory):
        self.memory = memory

    def execute(self, input_str: str) -> str:
        inputs = {self.PromptParam.INPUT: input_str}
        return self.execute_dict(inputs)
