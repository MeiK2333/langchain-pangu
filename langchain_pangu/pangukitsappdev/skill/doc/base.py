#  Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any

from langchain.schema.prompt_template import BasePromptTemplate

from langchain_pangu.pangukitsappdev.api.llms.base import LLMApi
from langchain_pangu.pangukitsappdev.api.memory.vector.base import Document
from langchain_pangu.pangukitsappdev.api.skill.base import Skill

logger = logging.getLogger(__name__)

DOCUMENTS_VAR_KEY = "documents"
QUESTION_VAR_KEY = "question"


class AbstractDocSkill(Skill, ABC):
    """文档处理Skill
    """

    @abstractmethod
    def execute_with_docs(self, docs: List[Document], question: str = None) -> str:
        """文档摘要/问答
        Args:
            docs: 文档
            question: (Optional)问题，如果不传递question，则对文档进行摘要，否则就是文档文档

        Returns:
            回答或者摘要
        """
        pass

    def execute_with_texts(self, texts: List[str], question: str = None) -> str:
        """文档摘要/问答
        Args:
            texts: 文档内容
            question: (Optional)问题，如果不传递question，则对文档进行摘要，否则就是文档文档

        Returns:
            回答或者摘要
        """
        return self.execute_with_docs([Document(page_content=t) for t in texts], question=question)

    def execute(self, inputs: Dict[str, Any]) -> str:
        return self.execute_with_docs(docs=inputs.get(DOCUMENTS_VAR_KEY), question=inputs.get(QUESTION_VAR_KEY))


class DocStuffSkill(AbstractDocSkill):
    """
    stuff策略
    """

    def __init__(self, prompt_template: BasePromptTemplate, llm_api: LLMApi):
        self.prompt_template = prompt_template
        self.llm_api = llm_api

    def execute_with_docs(self, docs: List[Document], question: str = None) -> str:
        prompt = self.prompt_template.format(**{DOCUMENTS_VAR_KEY: [{"pageContent": d.page_content} for d in docs],
                                                QUESTION_VAR_KEY: question})
        return self.skill_llm_ask(prompt, self.llm_api)


class DocRefineSkill(AbstractDocSkill):
    """
    refine策略

    Attributes:
        qa_prompt: 问答的prompt模板
        refine_prompt: 精炼的prompt模板
        llm_api: llm
    """

    def __init__(self, qa_prompt: BasePromptTemplate, refine_prompt: BasePromptTemplate, llm_api: LLMApi):
        self.qa_prompt = qa_prompt
        self.refine_prompt = refine_prompt
        self.llm_api = llm_api

    def execute_with_docs(self, docs: List[Document], question: str = None) -> str:
        prompt = self.qa_prompt.format(document=docs[0].page_content, question=question)
        answer = self.skill_llm_ask(prompt, self.llm_api)

        logger.debug("Refine policy first answer %s", answer)

        if len(docs) == 1:
            return answer

        for doc in docs[1:]:
            refine_prompt = self.refine_prompt.format(document=doc.page_content,
                                                      question=question,
                                                      answer=answer)
            answer = self.skill_llm_ask(refine_prompt, self.llm_api)
            logger.debug("Refine policy refined answer: %s", answer)

        return answer


class DocMapReduceSkill(AbstractDocSkill):
    """MapReduce策略
    Attributes:
        map_prompt: Map阶段使用的prompt模板
        reduce_prompt: reduce阶段使用的prompt模板
        llm_api: 大语言模型的接口
        reduce_max_token: （Optional） 默认每次组合reduce最大的token数量，防止超过LLM的参数限制
    """

    def __init__(self, map_prompt: BasePromptTemplate,
                 reduce_prompt: BasePromptTemplate,
                 llm_api: LLMApi,
                 reduce_max_token: int = 2000):
        self.map_prompt = map_prompt
        self.reduce_prompt = reduce_prompt
        self.llm_api = llm_api
        self.reduce_max_token = reduce_max_token

    def execute_with_docs(self, docs: List[Document], question: str = None) -> str:
        def map_doc(doc: Document) -> Document:
            prompt = self.map_prompt.format(document=doc.page_content, question=question)
            answer = self.skill_llm_ask(prompt, self.llm_api)
            logger.debug("policy: mapreduce, document is: %s", doc.page_content)
            logger.debug("policy: mapreduce, map answer is: %s", answer)
            return Document(page_content=answer)

        def format_reduce(docs_group: List[Document]) -> str:
            return self.reduce_prompt.format(summaries=[{"pageContent": d.page_content} for d in docs_group],
                                             question=question)

        def reduce_docs(docs_group: List[Document]) -> str:
            return self.skill_llm_ask(format_reduce(docs_group), self.llm_api)

        mapped_docs = [map_doc(doc) for doc in docs]

        while len(format_reduce(mapped_docs)) > self.reduce_max_token:
            docs_groups = []
            sub_docs = []
            for doc in mapped_docs:
                sub_docs.append(doc)
                if len(format_reduce(sub_docs)) < self.reduce_max_token:
                    continue

                if len(sub_docs) <= 2:
                    raise ValueError("single document is too long")

                docs_groups.append(sub_docs[0:-1])
                sub_docs.clear()
                sub_docs.append(doc)

            if sub_docs:
                docs_groups.append(sub_docs)

            mapped_docs.clear()

            mapped_docs = [Document(page_content=reduce_docs(docs_group)) for docs_group in docs_groups]

        return reduce_docs(mapped_docs)
