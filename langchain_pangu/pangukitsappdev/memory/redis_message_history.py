#  Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
from langchain.memory.chat_message_histories.redis import RedisChatMessageHistory

from langchain_pangu.pangukitsappdev.api.memory.cache.cache_config import ServerInfoRedis
from langchain_pangu.pangukitsappdev.api.memory.message_history_config import MessageHistoryConfig


class RedisMessageHistory(RedisChatMessageHistory):
    """继承 langchain RedisChatMessageHistory，根据 MessageHistoryConfig传参"""

    def __init__(self, msg_history_config: MessageHistoryConfig = None):
        if not msg_history_config:
            msg_history_config = MessageHistoryConfig()
        if not isinstance(msg_history_config.server_info, ServerInfoRedis):
            raise ValueError("Type of server_info should be ServerInfoRedis")
        super().__init__(
            session_id=msg_history_config.session_tag,
            url=msg_history_config.server_info.get_urls()[0],
            key_prefix=msg_history_config.key_prefix,
            ttl=msg_history_config.ttl
        )
