#  Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
import json
from typing import List

from langchain.schema import BaseChatMessageHistory
from langchain.schema.messages import BaseMessage, messages_from_dict
from sqlalchemy import Column, Integer, Text, create_engine, String
from sqlalchemy.orm import sessionmaker

from langchain_pangu.pangukitsappdev.api.memory.cache.cache_config import ServerInfoSql
from langchain_pangu.pangukitsappdev.api.memory.message_history_config import MessageHistoryConfig
from langchain_pangu.pangukitsappdev.utils.time_date import now_yyyyMMddHHmmss, now_sec, to_yyyyMMddHHmmss


def message_to_dict(message: BaseMessage) -> dict:
    return {"type": message.type, "data": message.dict()}


def create_message_model_with_create_date(table_name, dynamic_base):
    """
    基于表名创建 message model 对比 langchain SQLChatMessageHistory新增 create_date
    Args:
        table_name: The name of the table to use.
        dynamic_base: The base class to use for the model.
    Returns:
        The model class.
    """

    # Model declared inside a function to have a dynamic table name
    class Message(dynamic_base):
        __tablename__ = table_name
        id = Column(Integer, primary_key=True)
        session_tag = Column(Text)
        message = Column(Text)
        create_date = Column(String(45), nullable=True, default=None, index=True)

    return Message


class SQLMessageHistory(BaseChatMessageHistory):
    """SQLMessageHistory重写"""

    def __init__(self, msg_history_config: MessageHistoryConfig = None):
        if not msg_history_config:
            msg_history_config = MessageHistoryConfig(server_info=ServerInfoSql(env_prefix="sdk.memory.rds"))
        if not isinstance(msg_history_config.server_info, ServerInfoSql):
            raise ValueError("Type of server_info should be ServerInfoSql")
        self.table_name = msg_history_config.table_name
        self.connection_string = msg_history_config.server_info.get_urls()[0]
        self.engine = create_engine(self.connection_string, echo=False)
        self._create_table_if_not_exists()
        self.session_tag = msg_history_config.session_tag
        self.Session = sessionmaker(self.engine)
        self.ttl = msg_history_config.ttl

    def _create_table_if_not_exists(self) -> None:
        try:
            from sqlalchemy.orm import declarative_base
        except ImportError:
            from sqlalchemy.ext.declarative import declarative_base
        dynamic_base = declarative_base()
        self.Message = create_message_model_with_create_date(self.table_name, dynamic_base)
        # Create all does the check for us in case the table exists.
        dynamic_base.metadata.create_all(self.engine)

    @property
    def messages(self) -> List[BaseMessage]:
        """返回所有 session_tag下 message"""
        with self.Session() as session:
            result = session.query(self.Message).where(
                self.Message.session_tag == self.session_tag
            )
            items = [json.loads(record.message) for record in result]
            messages = messages_from_dict(items)
            return messages

    def add_message(self, message: BaseMessage) -> None:
        """新增 message 前移除过期 message"""
        self.evict()
        with self.Session() as session:
            json_str = json.dumps(message_to_dict(message))
            session.add(self.Message(
                session_tag=self.session_tag,
                message=json_str,
                create_date=now_yyyyMMddHHmmss()
            ))
            session.commit()

    def clear(self) -> None:
        """删除 session_tag 所有 message"""
        with self.Session() as session:
            session.query(self.Message).filter(
                self.Message.session_tag == self.session_tag
            ).delete()
            session.commit()

    def evict(self) -> None:
        """根据 ttl 删除过期 message"""
        with self.Session() as session:
            session.query(self.Message).filter(
                self.Message.session_tag == self.session_tag,
                self.Message.create_date < to_yyyyMMddHHmmss(now_sec() - self.ttl)
            ).delete()
            session.commit()
