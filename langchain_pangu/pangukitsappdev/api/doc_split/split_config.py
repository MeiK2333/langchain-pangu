#  Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
from typing import Optional

from pydantic.v1 import Field

from langchain_pangu.pangukitsappdev.api.common_config import IAMConfig, IAMConfigWrapper, HttpConfig
from langchain_pangu.pangukitsappdev.api.config_loader import SdkBaseSettings


class SplitConfig(SdkBaseSettings):
    """
    文档解析相关配置
    Attributes:
        css_url: 文档解析地址
        file_path: 文件路径
        file_name: 文件名
        mode: 拆分模式
        iam_config: 文档解析iam_config
        http_config: split使用http_config
    """

    css_url: Optional[str] = Field(env="sdk.doc.split.css.url")
    file_path: Optional[str] = Field(env="sdk.doc.split.css.filepath")
    file_name: Optional[str] = Field(env="sdk.doc.split.css.filename")
    mode: Optional[int] = Field(default=0, env="sdk.doc.split.css.mode")
    iam_config: IAMConfig = Field(default_factory=IAMConfigWrapper(env_prefix="sdk.doc.split.css.iam").get_iam_config)
    http_config: HttpConfig = Field(default_factory=lambda: HttpConfig(env_prefix="sdk.doc.split.css.proxy"))

    def upload_url(self):
        return self.css_url + '/doc-search/files'

    def result_url(self, task_id):
        return f'{self.css_url}/doc-search/tasks/{task_id}'
