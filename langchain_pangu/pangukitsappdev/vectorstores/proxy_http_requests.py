#  Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
from elasticsearch.connection.http_requests import RequestsHttpConnection


class ProxyRequestsHttpConnection(RequestsHttpConnection):
    def __init__(
            self,
            proxies=None,
            **kwargs
    ):
        """
        初始化
        Args:
            proxies: (Optional) 包含 proxies 参数，配置 es 代理信息
        """
        super(ProxyRequestsHttpConnection, self).__init__(
            **kwargs
        )
        if proxies:
            self.session.proxies = proxies
