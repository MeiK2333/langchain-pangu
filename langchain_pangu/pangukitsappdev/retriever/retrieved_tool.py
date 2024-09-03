#  Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
from langchain_pangu.pangukitsappdev.retriever.tool_metadata import ToolMetadata


class RetrievedTool(ToolMetadata):
    """
    工具的元数据信息
    Attributes:
        score: 评分，如从ES库检索后的score字段
    """
    score: float
