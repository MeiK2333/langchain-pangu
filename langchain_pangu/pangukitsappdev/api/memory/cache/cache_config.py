#  Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
from typing import Optional

from pydantic.v1 import Field

from langchain_pangu.pangukitsappdev.api.common_config import ServerInfo
from langchain_pangu.pangukitsappdev.api.config_loader import SdkBaseSettings
from langchain_pangu.pangukitsappdev.api.embeddings.base import EmbeddingApi


class ServerInfoRedis(ServerInfo):

    def has_auth(self) -> bool:
        return True if self.password else False

    def get_urls(self) -> [str]:
        if not self.url:
            return []

        from urllib.parse import quote
        result = []
        for u in self.url.split(","):
            if u.find('://') != -1:
                result.append(f"redis://:{quote(self.password)}@{u.split('://')[1]}/0"
                              if self.has_auth() else f"redis://{u.split('://')[1]}/0")
            else:
                result.append(f"redis://:{quote(self.password)}@{u}/0" if self.has_auth() else f"redis://{u}/0")

        return result


class ServerInfoSql(ServerInfo):
    """sql数据存储相关的配置
    Attributes:
        url: 拼接好的url； 如jdbc场景
        pool_size: 连接池大小
    """

    pool_size: Optional[str] = Field(env="poolSize", default=5)

    def get_urls(self) -> [str]:
        from sqlalchemy.engine.url import make_url
        url_instance = make_url(self.url)

        url_instance = url_instance.set(username=self.user, password=self.password)
        return [url_instance.render_as_string(False)]


class CacheStoreConfig(SdkBaseSettings):
    """
    缓存存储的相关配置
    Attributes:
        store_name: 存储名称
        server_info: ServerInfo，存储服务器连接相关信息，默认使用redis的连接
        embedding: embedding接口，语义缓存时需要使用
        vector_store_name: 向量语义缓存时，设置库名，默认使用css存储
        distance_strategy: 检索模式，默认使用内积inner_product
        score_threshold: 语义缓存， 相似度评分阈值；评分低于阈值表示不相似
        session_tag: 用户指定cache会话标志
        expire_after_access: 访问后到期时间，单位为秒， 默认不设置过期
        expire_after_write: 写入后到期时间，单位为秒， 默认不设置过期
        maximum_size: 基于个数大小
    """

    store_name: Optional[str]
    server_info: ServerInfo = Field(default_factory=lambda: ServerInfoRedis(env_prefix="sdk.memory.dcs"))
    embedding: Optional[EmbeddingApi]
    vector_store_name: str = Field(default="css")
    distance_strategy: str = Field(default="inner_product")
    score_threshold: float = Field(default=0.2)
    session_tag: str = Field(default="")
    expire_after_access: int = Field(default=-1)
    expire_after_write: int = Field(default=-1)
    maximum_size: int = Field(default=-1)
