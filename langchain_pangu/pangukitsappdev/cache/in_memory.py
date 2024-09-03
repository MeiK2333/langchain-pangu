#  Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
from typing import Any, Optional

from cachetools import TTLCache, LRUCache
from langchain.schema.cache import BaseCache
from langchain.schema.cache import RETURN_VAL_TYPE

from langchain_pangu.pangukitsappdev.api.memory.cache.base import CacheApiAdapter
from langchain_pangu.pangukitsappdev.api.memory.cache.cache_config import CacheStoreConfig


class InMemoryCacheApi(CacheApiAdapter):

    def __init__(self, cache_config: CacheStoreConfig):
        if cache_config.expire_after_write > 0:
            raise ValueError("InMemory Cache do not support expire_after_write")
        in_memory_cache = TTLInMemoryCache(cache_config.expire_after_access,
                                           cache_config.maximum_size)
        super().__init__(in_memory_cache, cache_config.session_tag)


class TTLInMemoryCache(BaseCache):
    def __init__(self, ttl, maximum_size):
        self.ttl = ttl
        max_size = maximum_size if maximum_size > 0 else float('inf')
        # 非Time-Based过期采用LRUCache
        if self.ttl <= 0:
            self.cache = LRUCache(maxsize=max_size)
        else:
            self.cache = TTLCache(maxsize=max_size, ttl=self.ttl)

    def lookup(self, prompt: str, llm_string: str) -> Optional[RETURN_VAL_TYPE]:
        return self.cache.get((prompt, llm_string))

    def update(self, prompt: str, llm_string: str, return_val: RETURN_VAL_TYPE) -> None:
        self.cache[(prompt, llm_string)] = return_val

    def clear(self, **kwargs: Any) -> None:
        self.cache.clear()
