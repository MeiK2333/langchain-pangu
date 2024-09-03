#  Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
import logging
from typing import Dict, Any, Optional, List
from uuid import UUID

from langchain.callbacks.tracers import ConsoleCallbackHandler
from langchain.schema import BaseMessage


class DebugConsoleCallbackHandler(ConsoleCallbackHandler):
    """
    Only for debug，打印解析后的prompt数据供调试使用
    """

    def on_chat_model_start(self, serialized: Dict[str, Any], messages: List[List[BaseMessage]], *, run_id: UUID,
                            parent_run_id: Optional[UUID] = None, **kwargs: Any) -> Any:
        pass

    def on_text(self, text: str, *, run_id: UUID, parent_run_id: Optional[UUID] = None, **kwargs: Any) -> Any:
        super().on_text(text, run_id=run_id, parent_run_id=parent_run_id, **kwargs)
        logging.INFO(text)
