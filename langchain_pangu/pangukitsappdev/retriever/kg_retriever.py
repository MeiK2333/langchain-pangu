#  Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
from typing import List

from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from langchain.schema import Document
from langchain.schema.retriever import BaseRetriever

from langchain_pangu.pangukitsappdev.utilities.knowledge_graph import KnowledgeGraph


class KGRetriever(BaseRetriever):
    kg: KnowledgeGraph

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
        return self.kg.query(query)
