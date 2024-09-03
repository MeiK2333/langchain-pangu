#  Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
from typing import Optional

from pydantic.v1 import Field

from langchain_pangu.pangukitsappdev.api.common_config import IAMConfig, OpenAIConfig, HttpConfig, IAMConfigWrapper
from langchain_pangu.pangukitsappdev.api.config_loader import SdkBaseSettings

DEFAULT_EMBEDDING_CONFIG_ENV_PREFIX = "sdk.embedding.css"


class EmbeddingConfig(SdkBaseSettings):

    def __init__(self, env_prefix=DEFAULT_EMBEDDING_CONFIG_ENV_PREFIX, **kwargs):
        super().__init__(env_prefix=env_prefix, **kwargs)

    embedding_name: Optional[str] = Field(default="default_embedding_name")
    css_url: Optional[str] = Field(env="url")
    iam_config: IAMConfig = Field(default_factory=IAMConfigWrapper(env_prefix="sdk.embedding.css.iam").get_iam_config)
    openai_config: OpenAIConfig = Field(default_factory=OpenAIConfig)
    http_config: HttpConfig = Field(default_factory=lambda: HttpConfig(env_prefix="sdk.embedding.css.proxy"))
