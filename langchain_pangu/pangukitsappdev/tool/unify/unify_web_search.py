#  Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
import json
import logging
from typing import Optional, List

from langchain_pangu.pangukitsappdev.retriever.web_retirever import WebRetriever
from langchain_pangu.pangukitsappdev.tool.tool import Tool
from langchain_pangu.pangukitsappdev.utilities.web_search import WebSearch

logger = logging.getLogger(__name__)


class UnifyWebSearch(Tool):

    name = "web_search"
    description = ""
    input_desc = ""
    output_desc = ""
    principle = ""
    web_search: Optional[WebSearch]
    pangu_function = '{"name": "web_search", "description:": "搜索引擎获取互联网知识",' \
                     ' "principle:": "问题可用搜索引擎辅助解决时使用",' \
                     ' "arguments": {"query": ["String:搜索query", "..."]}, "results": "搜索结果片段列表"}'

    def _run(self, query_list: List[str]) -> str:
        if self.web_search is None:
            raise ValueError("Param web_search must be specified")
        web_retriever = WebRetriever(web_search=self.web_search)
        query = " ".join(query_list)
        logger.info("[web_search工具调用问题]：%s", query)
        docs = web_retriever.get_relevant_documents(query=query, top_k=5)

        if len(docs) > 0:
            try:
                return json.dumps({"evidence": [{"category": doc.metadata.get("site_category", ["无"])[0],
                                                 "publish_time": doc.metadata.get("publish_time", "无"),
                                                 "text": doc.page_content,
                                                 "title": doc.metadata.get("title", "无")} for doc in docs],
                                   "recent_user_query": query}, ensure_ascii=False)
            except Exception as e:
                logger.error("出现如下异常%s", e)
                return "搜索信息错误，不要调用工具，直接回答问题。"

        else:
            return "搜索信息错误，不要调用工具，直接回答问题。"
