#  Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
from typing import Optional

from pydantic.v1 import BaseModel, Field, Extra

from langchain_pangu.pangukitsappdev.api.common_config import IAMConfig, HttpConfig, OpenAIConfig, IAMConfigWrapper
from langchain_pangu.pangukitsappdev.api.config_loader import SdkBaseSettings


class GalleryConfig(SdkBaseSettings):
    gallery_url: Optional[str] = Field(env="sdk.llm.gallery.url")
    iam_config: IAMConfig = Field(default_factory=IAMConfigWrapper(env_prefix="sdk.llm.gallery.iam").get_iam_config)
    http_config: HttpConfig = Field(default_factory=lambda: HttpConfig(env_prefix="sdk.llm.gallery.proxy"))


class LLMParamConfig(BaseModel):
    class Config:
        extra = Extra.forbid

    """
    The maximum number of tokens to generate in the completion.
    """
    max_tokens: Optional[int]

    """
    What sampling temperature to use, between 0 and 2. Higher values like 0.8 will make the output more random, 
    while lower values like 0.2 will make it more focused and deterministic. 
    We generally recommend altering this or top_p but not both.
    """
    temperature: Optional[float]

    """
    An alternative to sampling with temperature, called nucleus sampling, 
    where the model considers the results of the tokens with top_p probability mass. 
    So 0.1 means only the tokens comprising the top 10% probability mass are considered. 
    We generally recommend altering this or temperature but not both.
    """
    top_p: Optional[float]

    """
    How many completions to generate for each prompt. 
    Note: Because this parameter generates many completions, it can quickly consume your token quota. 
    Use carefully and ensure that you have reasonable settings for max_tokens and stop.
    """
    n: Optional[int]

    """
    Number between -2.0 and 2.0. Positive values penalize new tokens based on whether they appear in the text so far, 
    increasing the model's likelihood to talk about new topics.
    """
    presence_penalty: Optional[float]

    """
    Number between -2.0 and 2.0. 
    Positive values penalize new tokens based on their existing frequency in the text so far, 
    decreasing the model's likelihood to repeat the same line verbatim.
    """
    frequency_penalty: Optional[float]

    """
    Generates best_of completions server-side and returns the "best" (the one with the highest log probability\
     per token).
     Results cannot be streamed. When used with n, best_of controls the number of candidate completions and n specifies\
      how many to return – best_of must be greater than n. 
     Note: Because this parameter generates many completions, it can quickly consume your token quota. 
     Use carefully and ensure that you have reasonable settings for max_tokens and stop.
    """
    best_of: Optional[int]

    """
    If set, partial message deltas will be sent, like in ChatGPT. 
    Tokens will be sent as data-only server-sent events as they become available.
    """
    stream: Optional[bool]

    """
    是否由调用方提供完整prompt，可选参数，默认不设置
    """
    with_prompt: Optional[bool]


DEFAULT_LLM_MODULE_CONFIG_ENV_PREFIX = "sdk.llm.pangu"


class LLMModuleProperty(BaseModel):
    """
    Pangu Agent prompt标志
    Attributes:
        unify_tag_prefix: 输入prompt 起始占位符
        unify_tag_suffix: 输入prompt 结束占位符
        unify_tool_tag_prefix: 工具调用起始占位符
        unify_tool_tag_suffix: 工具调用结束占位符
    """
    unify_tag_prefix: Optional[str]
    unify_tag_suffix: Optional[str]
    unify_tool_tag_prefix: Optional[str]
    unify_tool_tag_suffix: Optional[str]


class LLMModuleConfig(SdkBaseSettings):
    """
    Pangu LLM的基本配置参数
    Attributes:
        llm_name: 模型名称
        url: 模型url
        system_prompt: 系统人设
        enable_append_system_message: 当设置了systemPrompt后，是否尝试自动添加一个SystemMessage，
        在Agent场景下，如果systemPrompt已经拼接在UserMessage了，则会设置为false，不再添加新的systemMessage
        module_version: 盘古模型版本
        cot_desc: cot描述
    """

    def __init__(self, env_prefix=DEFAULT_LLM_MODULE_CONFIG_ENV_PREFIX, **kwargs):
        super().__init__(env_prefix=env_prefix, **kwargs)

    llm_name: str = Field(default="pangu_llm")
    url: Optional[str] = Field(env="url")
    system_prompt: Optional[str]
    enable_append_system_message: bool = True
    module_version: Optional[str] = Field(env="model-version")
    llm_module_property: LLMModuleProperty = Field(default_factory=LLMModuleProperty)
    cot_desc: Optional[str]


class LLMConfig(SdkBaseSettings):
    """LLM参数

    Tips: 这里嵌套的对象，需要使用default_factory，而不是default
    Attributes:
        llm_module_config: llm站点配置，LLMModuleConfig
        iam_config: iam相关认证配置IAMConfig
        llm_param_config: iam认证相关配置LLMParamConfig，默认读取sdk.llm.iam开头的配置
        openai_config: openai认证，OpenAIConfig
        gallery_config: 第三方大模型站点配置，GalleryConfig
        http_config: http相关配置，HttpConfig
    """
    llm_module_config: LLMModuleConfig = Field(default_factory=LLMModuleConfig)
    iam_config: IAMConfig = Field(default_factory=IAMConfigWrapper(env_prefix="sdk.llm.pangu.iam").get_iam_config)
    llm_param_config: LLMParamConfig = Field(default_factory=LLMParamConfig)
    openai_config: OpenAIConfig = Field(default_factory=OpenAIConfig)
    gallery_config: GalleryConfig = Field(default_factory=GalleryConfig)
    http_config: HttpConfig = Field(default_factory=lambda: HttpConfig(env_prefix="sdk.llm.pangu.proxy"))
