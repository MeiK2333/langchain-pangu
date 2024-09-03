#  Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
from typing import Dict, Any, List

from langchain.memory.chat_memory import BaseChatMemory

from langchain_pangu.pangukitsappdev.api.skill.base import SimpleSkill


class ConversationSummaryBufferMemory(BaseChatMemory):
    """Buffer with summarizer for storing conversation memory."""
    buffer: str = ""
    summary_skill: SimpleSkill
    human_prefix: str = "Human"
    ai_prefix: str = "AI"
    memory_key: str = "history"

    @property
    def memory_variables(self) -> List[str]:
        """Will always return list of memory variables.

        :meta private:
        """
        return [self.memory_key]

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Return history buffer."""
        return {self.memory_key: self.buffer}

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """Save context from this conversation to buffer."""
        super().save_context(inputs, outputs)
        summary = {}
        summary.update(
            {"context": " ".join(["{}:{}".format(msg.type, msg.content) for msg in self.chat_memory.messages[-2:]])}
        )
        summary.update({"summary": self.buffer})
        self.buffer = self.summary_skill.execute(summary)

    def clear(self) -> None:
        """Clear memory contents."""
        super().clear()
        self.buffer = ""
