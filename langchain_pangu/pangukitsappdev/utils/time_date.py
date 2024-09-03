#  Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
import datetime
import time

F_yyyyMMddHHmmss = "%Y-%m-%d %H:%M:%S"

def now_millis() -> int:
    """
    返回当前的时间戳毫秒值
    :return: now毫秒值
    """
    return round(time.time() * 1000)


def now_sec() -> int:
    """返回当前时间戳，单位秒"""
    return int(time.time())


def now_yyyyMMddHHmmss() -> str:
    """
    以yyyy-MM-dd HH:mm:ss 格式返回当前的时间
    Returns:
        yyyy-MM-dd HH:mm:ss格式的时间

    """

    return datetime.datetime.now().strftime(F_yyyyMMddHHmmss)


def to_yyyyMMddHHmmss(ts_sec: int) -> str:
    """
    把ts_sec转换成yyyy-MM-dd HH:mm:ss 格式字符串
    Args:
        ts_sec: 秒表示的时间戳

    Returns:
        yyyy-MM-dd HH:mm:ss 格式时间戳

    """
    time_arr = time.localtime(ts_sec)
    return time.strftime(F_yyyyMMddHHmmss, time_arr)
