#  Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
import logging
from typing import List

import requests

from langchain_pangu.pangukitsappdev.api.memory.vector.base import Document
from langchain_pangu.pangukitsappdev.api.retriever.retriever_config import WebSearchConfig
from langchain_pangu.pangukitsappdev.utilities.web_search import WebSearch

logger = logging.getLogger(__name__)


class PetalSearch(WebSearch):

    def __init__(self, config: WebSearchConfig):
        """
        初始化
        :param config: 连接信息配置
        """
        self.config = config

    def query(self, query: str, top_k: int) -> List[Document]:
        resp = self.search(query, top_k)
        if resp is None:
            return []

        docs = [Document(page_content=web_page.get('content'),
                         metadata=web_page) for web_page in resp.get("web_pages")]
        return docs

    def search(self, query: str, top_k: int):
        """
        查询
        :param query: 查询问题
        :param top_k: 查询记录数
        :return: 相关网页信息
        """
        proxies = self.config.http_config.requests_proxies()
        requests_url = f"{self.config.server_info.url}?query={query}&limit={top_k}"
        # 账号密码配置
        headers = {"X-HW-ID": self.config.server_info.user,
                   "X-HW-APPKEY": self.config.server_info.password} \
            if self.config.server_info.password is not None else {}
        rsp = requests.get(requests_url, headers=headers, verify=False, proxies=proxies)
        if 200 == rsp.status_code:
            return rsp.json()
        else:
            logger.error("css petal search url error, http status: %d, error response: %s",
                         rsp.status_code, rsp.content)
            return None
