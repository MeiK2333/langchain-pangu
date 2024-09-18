import json
import os
from pathlib import Path
from typing import Union, Any

from langchain_pangu.pangukitsappdev.api.llms.llm_config import LLMConfig


class Utils:
    SSE_CONTINUE = 0
    SSE_DONE = 1
    SSE_DATA = 2
    SSE_EVENT = 3

    @staticmethod
    def sse_event(line: str) -> Union[int, Any]:
        line = line.strip()
        if not line:
            return Utils.SSE_CONTINUE, None
        if line.startswith("data:[DONE]"):
            return Utils.SSE_DONE, None
        if line.startswith("data:"):
            data_json = json.loads(line[5:])
            if data_json.get("choices") is None:
                raise ValueError(
                    f"Meet json decode error: %s, not get choices",
                    data_json,
                )
            return Utils.SSE_DATA, data_json["choices"][0]
        if line.startswith("event:"):
            return Utils.SSE_EVENT, json.loads(line[6:])
        return Utils.SSE_CONTINUE, None

    @staticmethod
    def set_kwargs(kw, **kwargs):
        for k, v in kwargs.items():
            if v is not None:
                kw[k] = v
        # 默认配置
        if "llm_config" not in kw:
            if "profile_file" in kw:
                os.environ["SDK_CONFIG_PATH"] = kw["profile_file"]
            elif Path("./llm.properties").exists():
                os.environ["SDK_CONFIG_PATH"] = "./llm.properties"
            kw["llm_config"] = LLMConfig()
            os.environ.pop("SDK_CONFIG_PATH", None)
        Utils.check_and_update_kwargs(**kw)

    @staticmethod
    def check_and_update_kwargs(**kwargs):
        """
        判断配置是否有效
        :param kwargs:
        :return:
        """
        llm_config: LLMConfig = kwargs["llm_config"]

        # 检查 pangu_url
        if "pangu_url" in kwargs:  # 优先获取参数配置
            llm_config.llm_module_config.url = kwargs["pangu_url"]
        elif llm_config.llm_module_config.url:
            pass
        else:
            raise ValueError("`pangu_url` field required")

        # 检查登录方式；aksk or user
        if "ak" in kwargs and "sk" in kwargs:
            llm_config.iam_config.iam_ak = kwargs["ak"]
            llm_config.iam_config.iam_sk = kwargs["sk"]
        elif "user" in kwargs and "password" in kwargs and "domain" in kwargs:
            llm_config.iam_config.iam_user = kwargs["user"]
            llm_config.iam_config.iam_pwd = kwargs["password"]
            llm_config.iam_config.iam_domain = kwargs["domain"]

        if llm_config.iam_config.iam_ak and llm_config.iam_config.iam_sk:
            pass
        elif (
            llm_config.iam_config.iam_user
            and llm_config.iam_config.iam_pwd
            and llm_config.iam_config.iam_domain
        ):
            pass
        else:
            raise ValueError(
                "`iam_ak` and `iam_sk` field required\n`iam_user` and `iam_pwd` and `iam_domain` field required"
            )

        if "model_version" in kwargs:
            llm_config.llm_module_config.module_version = kwargs["model_version"]
        if not llm_config.llm_module_config.module_version:
            raise ValueError("`model_version` field required")

        if "iam_url" in kwargs:
            llm_config.iam_config.iam_url = kwargs["iam_url"]
        elif llm_config.iam_config.iam_url:
            pass
        else:
            raise ValueError("`iam_url` field required")

        if "project" in kwargs:
            llm_config.iam_config.project_name = kwargs["project"]
        elif llm_config.iam_config.project_name:
            pass
        else:
            raise ValueError("`project` field required")
