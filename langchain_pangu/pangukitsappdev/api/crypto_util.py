#  Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
"""
SCC加解密工具类
Usage:
from crypto_util import crypto_util

"""

import logging
import os
import platform
import sys

logger = logging.getLogger(__name__)

SCC_CONF_PATH_ENV = "SCC_CONF_PATH"

SYSTEM_PLATFORM = platform.system().lower()
SCC_LIB_PATH = None
if SYSTEM_PLATFORM == 'windows':
    SCC_LIB_PATH = "C:/Program Files/SecComponent/lib"
elif SYSTEM_PLATFORM == "linux":
    SCC_LIB_PATH = "/usr/local/seccompnent/lib"

if SCC_LIB_PATH:
    sys.path.append(SCC_LIB_PATH)


class SccCryptoAPIWrapper:
    """SCC加解密Api
    需要在机器上安装华为云的scc安全组件后使用这个类。内部封装了CryptoAPI（scc组件安装后获的），在SccCryptoAPIWrapper新创建实例时会对
    CryptoAPI做初始化。

    Attributes:
        scc_conf_path: scc的配置路径
        _scc_api: 懒加载的CryptoAPI类实例，初始化的时候是None
    """

    def __init__(self, scc_conf_path=None):
        """
        Args:
            scc_conf_path: （Option）scc.conf的路径，不传递则从环境变量SCC_CONF_PATH获取
        """
        self._scc_api = None
        self.scc_conf_path = scc_conf_path

    def __del__(self):
        if self._scc_api:
            self._scc_api.finalize()

    def _init_scc(self):
        if self._scc_api:
            return

        conf_path = self.scc_conf_path if self.scc_conf_path else os.getenv(SCC_CONF_PATH_ENV)
        if not conf_path or not os.path.exists(conf_path):
            raise ValueError(
                "Please input scc_conf_path as params when __init__ "
                "or set env SCC_CONF_PATH and the conf file should exist!")

        try:
            from CryptoAPI import CryptoAPI
        except ImportError:
            raise ValueError(
                f"Can't import CryptoAPI, "
                f"please check scc has been installed and the {SYSTEM_PLATFORM}: {SCC_LIB_PATH} exists ")

        scc_api = CryptoAPI()
        scc_api.initialize(conf_path)
        if scc_api.gInitFlag:
            logger.info("SCC init success")
            self._scc_api = scc_api
        else:
            logger.error("SCC init failed, please check the logfile config in scc.conf")

    def encrypt(self, plain_text: str) -> str:
        """
        加密
        :param plain_text: 明文敏感信息
        :return: 加密后的密文
        """
        self._init_scc()
        return self._scc_api.encrypt(plain_text)

    def decrypt(self, cipher: str) -> str:
        """
        解密
        :param cipher: 密文
        :return: 解密后的明文
        """
        self._init_scc()
        return self._scc_api.decrypt(cipher)


crypto_util = SccCryptoAPIWrapper()


def decrypt(key_id: str, cipher: str) -> str:
    return crypto_util.decrypt(cipher)
