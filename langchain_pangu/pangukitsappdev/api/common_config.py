#  Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
from typing import Optional
from urllib.parse import quote

from pydantic.v1 import Field, validator

from langchain_pangu.pangukitsappdev.api.config_loader import SdkBaseSettings

AUTH_TOKEN_HEADER = "X-Auth-Token"
IAM_TOKEN_HEADER = "X-Subject-Token"


class HttpConfig(SdkBaseSettings):
    """http相关配置
    Attributes:
        proxy_default_enabled：默认代理开关
        proxy_enabled: 优先模块代理开关，模块未定义则使用默认代理开关
        proxy_url: 代理host地址:代理端口
        proxy_user: 认证用户
        proxy_passwd: 认证密码
    """

    def __init__(self, env_prefix="sdk.proxy", **kwargs):
        super().__init__(env_prefix=env_prefix, env_prefix_ident_fields=["enabled"], **kwargs)

    proxy_default_enabled: Optional[bool] = Field(env="sdk.proxy.enabled", default=False)
    proxy_enabled: Optional[bool] = Field(env="enabled")
    proxy_url: Optional[str] = Field(env="sdk.proxy.url")
    proxy_user: Optional[str] = Field(env="sdk.proxy.user")
    proxy_passwd: Optional[str] = Field(env="sdk.proxy.password")

    @validator("proxy_enabled")
    def get_proxy_enabled(cls, v, values):
        if v is None:
            v = values["proxy_default_enabled"]
        return v

    def get_proxy_url(self) -> str:
        if self.proxy_user and self.proxy_passwd:
            return f"{self.proxy_url.split('://')[0]}://{quote(self.proxy_user)}:{quote(self.proxy_passwd)}" \
                   f"@{self.proxy_url.split('://')[1]}"
        else:
            return f"{self.proxy_url}"

    def requests_proxies(self) -> dict:
        """
        构造requests请求时用的代理配置
        Returns:
            dict
        """
        return {
            "http": self.get_proxy_url(),
            "https": self.get_proxy_url(),
        } if self.proxy_enabled else {
            "http": "",
            "https": "",
        }


class IAMConfig(SdkBaseSettings):
    """
    IAM的配置参数
    """
    iam_disabled: Optional[bool] = Field(env="disabled")
    iam_ak: Optional[str] = Field(env="ak")
    iam_sk: Optional[str] = Field(env="sk")
    iam_url: Optional[str] = Field(env="url")
    iam_domain: Optional[str] = Field(env="domain")
    iam_user: Optional[str] = Field(env="user")
    iam_pwd: Optional[str] = Field(env="password")
    project_name: Optional[str] = Field(env="project")
    # token过期时间（毫秒），默认23小时，过期后，下一次接口调用会自动获取token
    expire_duration_millis: int = Field(default=23 * 60 * 60 * 1000, env="expireDurationMillis")
    x_auth_token: Optional[str] = Field(env="x_auth_token")
    http_config: HttpConfig = Field(default_factory=lambda: HttpConfig())


class IAMConfigWrapper:
    def __init__(self, env_prefix: str = "sdk.iam"):
        self.env_prefix = env_prefix

    def get_iam_config(self):
        sdk_iam = IAMConfig(env_prefix="sdk.iam",
                            http_config=HttpConfig(env_prefix=self.env_prefix.replace("iam", "proxy")))
        iam_config = IAMConfig(env_prefix=self.env_prefix,
                               http_config=HttpConfig(env_prefix=self.env_prefix.replace("iam", "proxy")))
        # 子模块设置iam_disabled使用，否则与全局iam_disabled一致
        if iam_config.iam_disabled is not None:
            sdk_iam.iam_disabled = iam_config.iam_disabled
        else:
            iam_config.iam_disabled = sdk_iam.iam_disabled
        if iam_config.iam_url is None and iam_config.x_auth_token is None:
            return sdk_iam
        else:
            return iam_config


class ServerInfo(SdkBaseSettings):
    """存储服务器链接相关信息
    Attributes:
        url: 主机ip列表:访问端口
        user: 用户
        password: 密钥
    """
    url: Optional[str] = Field(env="url")
    user: Optional[str] = Field(env="user")
    password: Optional[str] = Field(env="password")

    def has_auth(self) -> bool:
        return True if self.user and self.password else False

    def get_urls(self) -> [str]:
        raise ImportError("Not implemented")

    def get_http_urls(self) -> [str]:
        url_arr = self.url.split(",")
        return [
            f"{u.split('://')[0]}://{quote(self.user)}:{quote(self.password)}@{u.split('://')[1]}"
            for u in url_arr
        ] if self.has_auth() else [
            f"{u}" if 'http' in u else f"http://{u}"
            for u in url_arr
        ]


class OpenAIConfig(SdkBaseSettings):
    openai_base_url: Optional[str] = Field(env="sdk.llm.openai.url")
    openai_key: Optional[str] = Field(env="sdk.llm.openai.key")
    http_config: HttpConfig = Field(default_factory=lambda: HttpConfig(env_prefix="sdk.llm.openai.proxy"))
