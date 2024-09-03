#  Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
from typing import Dict, Type

from langchain_pangu.pangukitsappdev.api.memory.cache.base import CacheApi
from langchain_pangu.pangukitsappdev.api.memory.cache.cache_config import CacheStoreConfig, ServerInfoSql
from langchain_pangu.pangukitsappdev.cache.gptcache_adapter import SemanticGptCacheApi
from langchain_pangu.pangukitsappdev.cache.in_memory import InMemoryCacheApi
from langchain_pangu.pangukitsappdev.cache.sql import SqlCacheApi
from langchain_pangu.pangukitsappdev.cache.ttl_redis import RedisCacheApi, RedisSemanticCacheApi


class Caches:
    caches_map: Dict[str, Type[CacheApi]] = {}

    @classmethod
    def register(cls, cache_type: Type[CacheApi], cache_name: str):
        """注册一种cache的类型
        Args:
            cache_type: cache的类型，要求是CacheApi的子类
            cache_name: cache的名字，唯一代表这个cache的名字
        """
        cls.caches_map[cache_name] = cache_type

    @classmethod
    def of(cls, cache_name: str, cache_config: CacheStoreConfig = None) -> CacheApi:
        """根据名字创建一个CacheApi的实现类
        cache_name和cache_config.store_name不可同时为空
        Args:
            cache_name: （Optional）cache的名字，唯一标识一个CacheApi的实现类
            cache_config: （Optional）CacheApi的相关配置，如果不传递则从默认配置文件中或者环境变量中获取

        Returns:
            一个CacheApi的实现类

        """
        local_cache_name = cache_name if cache_name else cache_config.store_name

        if not local_cache_name:
            raise ValueError("The parameter cache_name or cache_config.store_name should have value")

        cache_type = cls.caches_map.get(local_cache_name)
        if not cache_type:
            raise ValueError(
                f"Unregistered cache name: {local_cache_name}, \
                    please call register(cache_type, cache_name) before use.")

        local_cache_config = cache_config if cache_config else cls._load_config(local_cache_name)
        if "sql" in local_cache_name and not isinstance(local_cache_config.server_info, ServerInfoSql):
            local_cache_config.server_info = ServerInfoSql(env_prefix="sdk.memory.rds")

        return cache_type(local_cache_config)

    @classmethod
    def _load_config(cls, local_cache_name):
        if "sql" in local_cache_name:
            server_info = ServerInfoSql(env_prefix="sdk.memory.rds")
            return CacheStoreConfig(server_info=server_info)

        return CacheStoreConfig()


Caches.register(RedisCacheApi, "redis")
Caches.register(RedisSemanticCacheApi, "semantic_redis")
Caches.register(InMemoryCacheApi, "inMemory")
Caches.register(SemanticGptCacheApi, "semantic_gptcache")
Caches.register(SqlCacheApi, "sql")
