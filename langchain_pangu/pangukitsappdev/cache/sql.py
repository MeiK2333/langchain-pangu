#  Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
from typing import Any, Optional

from langchain.cache import SQLAlchemyCache
from langchain.schema import Generation
from langchain.schema.cache import RETURN_VAL_TYPE
from sqlalchemy import Column, Integer, Text, String, create_engine, Engine, select, delete, make_url
from sqlalchemy.orm import declarative_base, Session

from langchain_pangu.pangukitsappdev.api.memory.cache.base import CacheApiAdapter
from langchain_pangu.pangukitsappdev.api.memory.cache.cache_config import CacheStoreConfig
from langchain_pangu.pangukitsappdev.utils.time_date import now_yyyyMMddHHmmss, now_sec, to_yyyyMMddHHmmss

PROMPT_PREFIX_LEN = 250

Base = declarative_base()

SQL_CACHE_TABLE_NAME = "tbl_full_llm_cache_py"


class MySQLLLMCacheSchema(Base):
    """MySQL table for full LLM Cache (all generations)."""

    __tablename__ = SQL_CACHE_TABLE_NAME
    id = Column(Integer, primary_key=True, autoincrement=True)
    prompt = Column(Text())
    prompt_prefix = Column(String(765), nullable=True, default=None, index=True)
    session_tag = Column(String(511), nullable=True, default=None, index=True)
    answer = Column(Text(), nullable=True)
    create_date = Column(String(45), nullable=True, default=None, index=True)


SCHEMA_LOOKUP = {
    "mysql": MySQLLLMCacheSchema
}


class SqlCacheApi(CacheApiAdapter):

    def __init__(self, cache_config: CacheStoreConfig):
        if cache_config.expire_after_access > 0:
            raise ValueError("Sql Cache do not support expire_after_access")
        url_instance = make_url(cache_config.server_info.get_urls()[0])
        backend_engine = url_instance.get_backend_name()
        """默认使用mysql"""
        schema_clz = SCHEMA_LOOKUP.get(backend_engine, MySQLLLMCacheSchema)
        sql_alchemy_cache = SQLCache(engine=create_engine(url_instance),
                                     cache_schema=schema_clz,
                                     ttl=cache_config.expire_after_write,
                                     maximum_size=cache_config.maximum_size,
                                     pool_size=cache_config.server_info.pool_size)
        super().__init__(sql_alchemy_cache, cache_config.session_tag)


class SQLCache(SQLAlchemyCache):

    def __init__(self, engine: Engine, cache_schema, ttl, maximum_size, pool_size=5):
        super().__init__(engine, cache_schema)
        self.ttl = ttl
        self.maximum_size = maximum_size if maximum_size > 0 else float('inf')

    def lookup(self, prompt: str, llm_string: str) -> Optional[RETURN_VAL_TYPE]:
        self.evict(prompt, llm_string)

        stmt = (
            select(self.cache_schema.answer)
            .where(self.cache_schema.session_tag == llm_string)
            .where(self.cache_schema.prompt_prefix == prompt[:PROMPT_PREFIX_LEN])
            .where(self.cache_schema.prompt == prompt)
        )
        with Session(self.engine) as session:
            rows = session.execute(stmt).fetchall()
            if rows:
                return [Generation(text=row[0]) for row in rows]
        return None

    def update(self, prompt: str, llm_string: str, return_val: RETURN_VAL_TYPE) -> None:
        items = [
            self.cache_schema(prompt=prompt,
                              prompt_prefix=prompt[:PROMPT_PREFIX_LEN],
                              session_tag=llm_string, answer=gen.text, create_date=now_yyyyMMddHHmmss())
            for i, gen in enumerate(return_val)
        ]
        with Session(self.engine) as session, session.begin():
            for item in items:
                session.merge(item)
        self.delete_by_batch(llm_string)

    def clear(self, **kwargs: Any) -> None:
        session_tag = kwargs.get("llm_string")

        if session_tag:
            with Session(self.engine) as session:
                session.query(self.cache_schema).where(self.cache_schema.session_tag == session_tag).delete()
                session.commit()
            return

        with Session(self.engine) as session:
            session.query(self.cache_schema).delete()
            session.commit()

    def evict(self, prompt, session_tag):
        if self.ttl <= 0:
            return
        del_stmt = (
            delete(self.cache_schema)
            .where(self.cache_schema.session_tag == session_tag)
            .where(self.cache_schema.prompt_prefix == prompt[:PROMPT_PREFIX_LEN])
            .where(self.cache_schema.create_date < to_yyyyMMddHHmmss(now_sec() - self.ttl))
            .where(self.cache_schema.prompt == prompt)
        )

        with Session(self.engine) as session:
            session.execute(del_stmt)
            session.commit()

    def delete_by_batch(self, llm_string: str):
        with Session(self.engine) as session:
            records = session.query(self.cache_schema)\
                .where(self.cache_schema.session_tag == llm_string).order_by(self.cache_schema.id).all()
            count = len(records)
            if count > self.maximum_size:
                for i in range(count - self.maximum_size):
                    session.delete(records[i])
                session.commit()

