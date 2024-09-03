#  Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
import logging
from typing import List

import requests

from langchain_pangu.pangukitsappdev.api.memory.vector.base import Document
from langchain_pangu.pangukitsappdev.api.retriever.retriever_config import KGConfig
from langchain_pangu.pangukitsappdev.utilities.knowledge_graph import KnowledgeGraph
from langchain_pangu.pangukitsappdev.utils.security_util import SecurityUtil
from langchain_pangu.pangukitsappdev.utils.time_date import now_millis

logger = logging.getLogger(__name__)


class KG(KnowledgeGraph):
    def __init__(self, config: KGConfig):
        """
        初始化
        :param config: 连接信息配置
        """
        self.config = config

    def query(self, query: str) -> List[Document]:
        resp = self.search(query)
        if resp is None:
            return []

        docs = [Document(page_content=resp.get('avoice'), metadata=resp)]
        return docs

    def search(self, query: str):
        """
        查询
        :param query: 查询问题
        :return: 对应结果
        """
        requests_body = {
            "queryString": query,
            "locale": "zh-cn"
        }
        logger.debug("css kg request: %s", str(requests_body))
        proxies = self.config.http_config.requests_proxies()
        headers = self.set_auth()
        headers.update({"Content-Type": "application/json"})
        rsp = requests.post(url=self.config.server_info.url,
                            json=requests_body,
                            headers=headers,
                            verify=False,
                            proxies=proxies)

        logger.debug("css kg request: %s", str(rsp))
        if 200 == rsp.status_code:
            return rsp.json()
        else:
            logger.error("css petal search url error, http status: %d, error response: %s",
                         rsp.status_code, rsp.content)
            return None

    def set_auth(self) -> dict:
        if self.config.server_info.password is None:
            return {}
        nonce = SecurityUtil.get_security_random()
        timestamp = now_millis()
        data = str(timestamp) + nonce
        auth = SecurityUtil.hmac_sha256_base64(data, self.config.server_info.password)
        return {
            "nonce": nonce,
            "timestamp": str(timestamp),
            "AuthCode": auth
        }
