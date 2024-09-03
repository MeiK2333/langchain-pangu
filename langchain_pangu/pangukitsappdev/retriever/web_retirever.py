#  Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
from typing import List

from langchain.schema import BaseRetriever, Document

from langchain_pangu.pangukitsappdev.utilities.web_search import WebSearch


class WebRetriever(BaseRetriever):
    web_search: WebSearch

    def _get_relevant_documents(self, query: str, top_k=5) -> List[Document]:
        return self.web_search.query(query, top_k)
