#  Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
import json
from typing import Dict

import requests

from langchain_pangu.pangukitsappdev.api.common_config import IAMConfig
from langchain_pangu.pangukitsappdev.api.common_config import IAM_TOKEN_HEADER
from langchain_pangu.pangukitsappdev.utils.time_date import now_millis


class IAMTokenProvider:
    """盘古鉴权的接口
        token = PanGuAuth(url={iam auth url}, username={your username}, password={your password},
        domain_name={your domain_name},project_name={your project_name} )
    """

    def __init__(self, iam_config: IAMConfig) -> None:

        """构造器

        :param iam_config: 相关配置
        """
        self.iam_disabled = iam_config.iam_disabled
        self.iam_ak = iam_config.iam_ak
        self.iam_sk = iam_config.iam_sk
        self.iam_url = iam_config.iam_url
        self.project_name = iam_config.project_name
        self.iam_domain = iam_config.iam_domain
        self.iam_pwd = iam_config.iam_pwd
        self.iam_user = iam_config.iam_user
        self.expire_duration_millis = iam_config.expire_duration_millis
        self.proxies = iam_config.http_config.requests_proxies()
        self.x_auth_token = iam_config.x_auth_token

        # 当前的token，可能过期了
        self.__hold_token: str = ""

        # 在什么时间超时，一个时间戳
        self.__expired_at: int = -1
        # 什么时间创建的
        self.__created_at: int = -1

    @staticmethod
    def get_token(url: str, username: str, password: str, domain_name: str, project_name: str, proxies: dict) -> str:
        data = {
            'auth': {
                'identity': {
                    'methods': ['password'],
                    'password': {
                        'user': {
                            'name': username,
                            'password': password,
                            'domain': {
                                'name': domain_name
                            }
                        }
                    }
                },
                'scope': {
                    'project': {
                        'name': project_name
                    }
                }
            }
        }
        headers = {
            'Content-Type': 'application/json'
        }

        response = requests.post(url, data=json.dumps(data), headers=headers, verify=False, timeout=15, proxies=proxies)
        if response.status_code == 201:
            # 请求成功，处理响应数据
            json_response = response.headers
            token = json_response[IAM_TOKEN_HEADER]
            return token

        response.raise_for_status()

    @staticmethod
    def get_token_by_ak_sk(url: str, access_key: str, secret_key: str, project_name: str, proxies: dict) -> str:
        data = {
            'auth': {
                'identity': {
                    'hw_ak_sk': {
                        'access': {
                            'key': access_key
                        },
                        "secret": {
                            'key': secret_key
                        }
                    },
                    'methods': ['hw_ak_sk']
                },
                'scope': {
                    'project': {
                        'name': project_name
                    }
                }
            }
        }
        headers = {
            'Content-Type': 'application/json'
        }

        response = requests.post(url, data=json.dumps(data), headers=headers, verify=False, timeout=15, proxies=proxies)
        if response.status_code == 201:
            # 请求成功，处理响应数据
            json_response = response.headers
            token = json_response[IAM_TOKEN_HEADER]
            return token

        response.raise_for_status()
        return ""

    def get_valid_token(self):
        """获取有效token
        使用这个方法获取可用token，并在到期后自动刷新
        :return: iam认证token
        """
        if self.iam_disabled:
            return ""

        if self.x_auth_token:
            return self.x_auth_token

        if self.__expired_at and now_millis() > self.__expired_at:
            if self.iam_sk is not None and self.iam_ak is not None:
                fresh_token = IAMTokenProvider.get_token_by_ak_sk(url=self.iam_url,
                                                                  access_key=self.iam_ak,
                                                                  secret_key=self.iam_sk,
                                                                  project_name=self.project_name,
                                                                  proxies=self.proxies)
            else:
                fresh_token = IAMTokenProvider.get_token(url=self.iam_url,
                                                         username=self.iam_user,
                                                         password=self.iam_pwd,
                                                         domain_name=self.iam_domain,
                                                         project_name=self.project_name,
                                                         proxies=self.proxies)
            self._update_token(fresh_token)

        return self.__hold_token

    def _update_token(self, fresh_token):
        self.__hold_token = fresh_token
        self.__created_at = now_millis()
        self.__expired_at = self.__created_at + self.expire_duration_millis


class IAMTokenProviderFactory:
    # 实现享元模式，同样的配置不用每次都创建一次。key是iamconfig的序列化数据
    _instance_cache: Dict[str, IAMTokenProvider] = {}

    @classmethod
    def create(cls, iam_config: IAMConfig) -> IAMTokenProvider:
        config_json_str = iam_config.json()
        cached = cls._instance_cache.get(config_json_str)
        if cached:
            return cached
        else:
            new_one = IAMTokenProvider(iam_config)
            # 更新缓存
            cls._instance_cache[config_json_str] = new_one

            return new_one
