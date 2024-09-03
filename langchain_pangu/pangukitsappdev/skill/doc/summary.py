#  Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
from langchain_pangu.pangukitsappdev.api.llms.base import LLMApi
from langchain_pangu.pangukitsappdev.prompt.prompt_tmpl import PromptTemplates as pt
from langchain_pangu.pangukitsappdev.skill.doc.base import DocStuffSkill, DocRefineSkill, DocMapReduceSkill


class DocSummaryStuffSkill(DocStuffSkill):
    """文档摘要stuff策略

    """

    def __init__(self, llm_api: LLMApi):
        super().__init__(pt.get("documents_summarize_stuff"), llm_api)


class DocSummaryRefineSkill(DocRefineSkill):
    """文档摘要refine策略"""

    def __init__(self, llm_api: LLMApi):
        super().__init__(pt.get("documents_summarize_refine_qa"), pt.get("documents_summarize_refine_combine"), llm_api)


class DocSummaryMapReduceSkill(DocMapReduceSkill):
    """文档问答MapReduce策略"""

    def __init__(self, llm_api: LLMApi, reduce_max_token: int = 2000):
        super().__init__(pt.get("documents_summarize_mapreduce_map"),
                         pt.get("documents_summarize_mapreduce_reduce"), llm_api, reduce_max_token)
