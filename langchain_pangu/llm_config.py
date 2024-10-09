import os
from typing import Optional
from urllib.parse import quote

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


def env_file_func():
    yield os.environ.get("SDK_CONFIG_PATH", ".env")


class HttpConfig(BaseSettings):
    proxy_default_enabled: Optional[bool] = Field(
        alias="sdk.proxy.enabled", default=False
    )
    proxy_enabled: Optional[bool] = Field(None, alias="enabled")
    proxy_url: Optional[str] = Field(None, alias="sdk.proxy.url")
    proxy_user: Optional[str] = Field(None, alias="sdk.proxy.user")
    proxy_passwd: Optional[str] = Field(None, alias="sdk.proxy.password")

    class Config:
        env_file = env_file_func()
        extra = "allow"

    def get_proxy_url(self) -> str:
        if self.proxy_user and self.proxy_passwd:
            return (
                f"{self.proxy_url.split('://')[0]}://{quote(self.proxy_user)}:{quote(self.proxy_passwd)}"
                f"@{self.proxy_url.split('://')[1]}"
            )
        else:
            return f"{self.proxy_url}"

    def requests_proxies(self) -> dict:
        """
        构造requests请求时用的代理配置
        Returns:
            dict
        """
        return (
            {
                "http": self.get_proxy_url(),
                "https": self.get_proxy_url(),
            }
            if self.proxy_enabled
            else {
                "http": "",
                "https": "",
            }
        )


class LLMModuleProperty(BaseModel):
    unify_tag_prefix: Optional[str] = Field(None)
    unify_tag_suffix: Optional[str] = Field(None)
    unify_tool_tag_prefix: Optional[str] = Field(None)
    unify_tool_tag_suffix: Optional[str] = Field(None)


class LLMModuleConfig(BaseSettings):
    url: Optional[str] = Field(None, alias="sdk.llm.pangu.url")
    module_version: Optional[str] = Field(None, alias="sdk.llm.pangu.model-version")
    llm_module_property: LLMModuleProperty = Field(default_factory=LLMModuleProperty)
    system_prompt: Optional[str] = Field(None)
    cot_desc: Optional[str] = Field(None)

    class Config:
        env_file = env_file_func()
        extra = "allow"


class IAMConfig(BaseSettings):
    iam_disabled: bool = Field(False, alias="sdk.iam.disabled")
    iam_ak: Optional[str] = Field(None, alias="sdk.iam.ak")
    iam_sk: Optional[str] = Field(None, alias="sdk.iam.sk")
    iam_url: Optional[str] = Field(None, alias="sdk.iam.url")
    iam_domain: Optional[str] = Field(None, alias="sdk.iam.domain")
    iam_user: Optional[str] = Field(None, alias="sdk.iam.user")
    iam_pwd: Optional[str] = Field(None, alias="sdk.iam.password")
    project_name: Optional[str] = Field(None, alias="sdk.iam.project")
    expire_duration_millis: int = Field(
        default=23 * 60 * 60 * 1000, alias="sdk.iam.expireDurationMillis"
    )
    http_config: HttpConfig = Field(default_factory=HttpConfig)
    x_auth_token: Optional[str] = Field(None, alias="sdk.iam.x_auth_token")

    class Config:
        env_file = env_file_func()
        extra = "allow"


class LLMParamConfig(BaseModel):
    max_tokens: Optional[int] = Field(None)
    temperature: Optional[float] = Field(None)
    top_p: Optional[float] = Field(None)
    n: Optional[int] = Field(None)
    presence_penalty: Optional[float] = Field(None)
    frequency_penalty: Optional[float] = Field(None)
    best_of: Optional[int] = Field(None)
    stream: Optional[bool] = Field(None)
    with_prompt: Optional[bool] = Field(None)


class LLMConfig(BaseModel):
    llm_module_config: LLMModuleConfig = Field(default_factory=LLMModuleConfig)
    iam_config: IAMConfig = Field(default_factory=IAMConfig)
    llm_param_config: LLMParamConfig = Field(default_factory=LLMParamConfig)
    http_config: HttpConfig = Field(default_factory=HttpConfig)


if __name__ == "__main__":
    llm_config = LLMConfig()
    print(llm_config)
