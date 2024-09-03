#  Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
from abc import ABC, abstractmethod
from typing import Optional

from langchain.schema import Generation
from langchain.schema.cache import BaseCache

from langchain_pangu.pangukitsappdev.api.schema import LLMResp


class CacheApi(ABC):
    """SDK封装的缓存对外接口"""

    @abstractmethod
    def lookup(self, prompt: str) -> LLMResp:
        """查询

        Args:
            prompt: 提示词

        Returns: 查询结果

        """

    @abstractmethod
    def update(self, prompt: str, value: LLMResp) -> None:
        """
        更新
        Args:
            prompt: 提示词
            value: 回答结果

        Returns:
            LLMResp
        """

    @abstractmethod
    def clear(self) -> None:
        """
        清空
        Args:

        Returns:
            LLMResp
        """


class CacheApiAdapter(CacheApi):
    """对Langchain原生的缓存做适配

    Attributes:
        cache: BaseCache的实现类，封装的langchain的cache
    """

    def __init__(self, cache: BaseCache, session_tag: str = ""):
        self.cache = cache
        self.session_tag = session_tag

    def lookup(self, prompt: str) -> Optional[LLMResp]:
        cached_llm_result = self.cache.lookup(prompt=prompt, llm_string=self.session_tag)
        if isinstance(cached_llm_result, list) and cached_llm_result:
            answer = cached_llm_result[0].text
            return LLMResp(answer=answer, is_from_cache=True)
        else:
            return None

    def update(self, prompt: str, value: LLMResp) -> None:
        self.cache.update(prompt=prompt,
                          llm_string=self.session_tag,
                          return_val=[Generation(text=value.answer)])

    def clear(self) -> None:
        self.cache.clear(llm_string=self.session_tag)
