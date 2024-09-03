#  Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
from typing import Any

from langchain.cache import RedisCache, RedisSemanticCache
from langchain.embeddings.base import Embeddings
from langchain.schema.cache import RETURN_VAL_TYPE
from redis import Redis

from langchain_pangu.pangukitsappdev.api.memory.cache.base import CacheApiAdapter
from langchain_pangu.pangukitsappdev.api.memory.cache.cache_config import CacheStoreConfig


class RedisCacheApi(CacheApiAdapter):
    """支持TTL的缓存策略，内部封装Langchain的redis缓存
    1. 通过CacheStoreConfig创建Redis实例，从而初始化RedisCache\n
    """

    def __init__(self, cache_config: CacheStoreConfig):
        if cache_config.expire_after_access > 0:
            raise ValueError("Redis Cache do not support expire_after_access")
        if cache_config.maximum_size > 0:
            raise ValueError("Redis Cache do not support maximum_size")
        redis_client: Redis = Redis.from_url(cache_config.server_info.get_urls()[0])

        redis_cache = TTLRedisCache(redis_client,
                                    cache_config.expire_after_write)
        super().__init__(redis_cache, cache_config.session_tag)


class TTLRedisCache(RedisCache):
    """支持TTL的缓存策略，继承自langchain的BaseCache
    """

    def __init__(self, redis_client, ttl):
        super().__init__(redis_=redis_client)
        self.expire_after_write = ttl

    def update(self, prompt: str, llm_string: str, return_val: RETURN_VAL_TYPE) -> None:
        super().update(prompt, llm_string, return_val)
        if self.expire_after_write > 0:
            self.redis.expire(name=self._key(prompt, llm_string), time=self.expire_after_write)

    """重写clear方法，统一CacheApiAdapter入参接口
    """
    def clear(self, **kwargs: Any):
        super().clear()


class RedisSemanticCacheApi(CacheApiAdapter):
    """Redis的语义缓存
    内部封装pangukitsappdev.cache.redis_ext.TTLRedisSemanticCache。支持配置缓存的ttl
    """

    def __init__(self, cache_config: CacheStoreConfig):
        config = {
            "redis_url": cache_config.server_info.get_urls()[0],
            "embedding": cache_config.embedding,
            "score_threshold": cache_config.score_threshold,
        }
        # 目前langchain不支持RedisSemanticCache，ttl功能与langchain同步，无需锁定版本
        if "ttl" in RedisSemanticCache.__init__.__code__.co_varnames and cache_config.expire_after_write > 0:
            config.update({"ttl": cache_config.expire_after_write})

        redis_semantic_cache = RedisSemanticCache(**config)
        super().__init__(redis_semantic_cache, cache_config.session_tag)


class TTLRedisSemanticCache(RedisSemanticCache):
    def __init__(self, redis_url: str,
                 embedding: Embeddings,
                 ttl,
                 score_threshold: float = 0.2,
                 ):
        super().__init__(redis_url, embedding, score_threshold)
        self.ttl = ttl

    def update(self, prompt: str, llm_string: str, return_val: RETURN_VAL_TYPE) -> None:
        """更新缓存，并配置缓存的ttl
        逻辑复用父类逻辑，获取ids进行ttl配置
        Args:
            prompt: prompt query
            llm_string: 唯一标识llm
            return_val: llm返回的数据
        """
        llm_cache = self._get_llm_cache(llm_string)
        # Write to vectorstore
        metadata = {
            "llm_string": llm_string,
            "prompt": prompt,
            "return_val": [generation.text for generation in return_val],
        }
        keys = llm_cache.add_texts(texts=[prompt], metadatas=[metadata])
        llm_cache.client.expire(name=keys[0], time=self.ttl)
