#  Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
from typing import Optional

from pydantic.v1 import Field

from langchain_pangu.pangukitsappdev.api.common_config import ServerInfo
from langchain_pangu.pangukitsappdev.api.config_loader import SdkBaseSettings
from langchain_pangu.pangukitsappdev.api.memory.cache.cache_config import ServerInfoRedis


class MessageHistoryConfig(SdkBaseSettings):
    """
    缓存存储的相关配置
    Attributes:
        store_name: 存储名称
        server_info: ServerInfo，存储服务器连接相关信息，默认使用redis的连接
        key_prefix: redis key前缀
        table_name: sql 表名称
        ttl: 缓存的老化时间，单位 秒
        session_tag: 用户指定history会话标志
    """

    store_name: Optional[str]
    server_info: ServerInfo = Field(default_factory=lambda: ServerInfoRedis(env_prefix="sdk.memory.dcs"))
    key_prefix: str = Field(default="message_store:")
    table_name: str = Field("tbl_chat_message_history")
    ttl: int = Field(default=24 * 3600)
    session_tag: str = Field(default="")
