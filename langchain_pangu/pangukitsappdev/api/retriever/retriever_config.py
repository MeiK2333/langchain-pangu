#  Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
from pydantic.v1 import Field

from langchain_pangu.pangukitsappdev.api.common_config import HttpConfig, ServerInfo
from langchain_pangu.pangukitsappdev.api.config_loader import SdkBaseSettings


class KGConfig(SdkBaseSettings):
    server_info: ServerInfo = Field(default_factory=lambda: ServerInfo(env_prefix="sdk.retriever.kg"))
    http_config: HttpConfig = Field(default_factory=lambda: HttpConfig(env_prefix="sdk.retriever.kg.proxy"))


class WebSearchConfig(SdkBaseSettings):
    server_info: ServerInfo = Field(default_factory=lambda: ServerInfo(env_prefix="sdk.retriever.petalSearch"))
    http_config: HttpConfig = Field(default_factory=lambda: HttpConfig(env_prefix="sdk.retriever.petalSearch.proxy"))
