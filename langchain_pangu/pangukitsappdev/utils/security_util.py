#  Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
import base64
import hmac
import secrets
from hashlib import sha256


class SecurityUtil:

    @staticmethod
    def get_security_random(bits: int = 256) -> str:
        """
        返回当前的时间戳毫秒值
        :param bits: 随机数的bit位数（取8的倍数，如256）
        :return: 十六进制字符串
        """
        try:
            return str(secrets.token_bytes(int(bits / 8)).hex())
        except ValueError:
            raise ValueError("get secure random exception.")

    @staticmethod
    def hmac_sha256_base64(msg: str, key_str: str) -> str:
        """
        进行hmacSHA256的加密，返回base64的字符串
        :param msg: 加密明文
        :param key_str: 密钥,经过HEX编码后的String串
        :return: Hex后的字符串
        """
        hmac_msg = msg.encode('utf-8')
        hmac_key = key_str.encode('utf-8')
        signature = base64.b64encode(hmac.new(hmac_key, hmac_msg, digestmod=sha256).digest()).decode('utf-8')
        return signature
