#  Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
from pangukitsappdev.api.config_loader import SdkBaseSettings
from pydantic import Field

from pangukitsappdev.api.common_config import HttpConfig, ServerInfo


class KGConfig(SdkBaseSettings):
    server_info: ServerInfo = Field(default_factory=lambda: ServerInfo(env_prefix="sdk.retriever.kg"))
    http_config: HttpConfig = Field(default_factory=lambda: HttpConfig(env_prefix="sdk.retriever.kg.proxy"))


class WebSearchConfig(SdkBaseSettings):
    server_info: ServerInfo = Field(default_factory=lambda: ServerInfo(env_prefix="sdk.retriever.petalSearch"))
    http_config: HttpConfig = Field(default_factory=lambda: HttpConfig(env_prefix="sdk.retriever.petalSearch.proxy"))
