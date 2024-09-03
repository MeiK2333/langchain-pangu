#  Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.

import time
from typing import List

import requests

from langchain_pangu.pangukitsappdev.api.common_config import AUTH_TOKEN_HEADER
from langchain_pangu.pangukitsappdev.api.doc_split.base import AbstractLoaderApi
from langchain_pangu.pangukitsappdev.api.doc_split.split_config import SplitConfig
from langchain_pangu.pangukitsappdev.api.memory.vector.base import Document
from langchain_pangu.pangukitsappdev.auth.iam import IAMTokenProviderFactory


def extract_actions(result):
    paragraphs = []
    for page in result['pages']:
        page_num = page['page_num']
        for component in page['components']:
            paragraphs.append(
                Document(
                    page_content=component.get("text", ""),
                    metadata={
                        "id": component['id'],
                        "title": component.get("title", ""),
                        "component_num": component["component_num"],
                        "page_num": page_num,
                        "url": result['doc_name']
                    }))
    return paragraphs


class DocPanguSplit(AbstractLoaderApi):

    def __init__(self, split_config: SplitConfig):
        super().__init__(split_config)
        self.token_provider = IAMTokenProviderFactory.create(self.split_config.iam_config)
        self.proxies = self.split_config.http_config.requests_proxies()

    """
    文档解析，目前支持doc/pdf
    """

    def load(self) -> List[Document]:
        token = self.token_provider.get_valid_token()

        headers = {
            AUTH_TOKEN_HEADER: token
        } if token else {}

        task_id = self.submit_task(headers, self.split_config.mode)
        result = self.get_task_result(task_id, headers)
        docs = extract_actions(result)
        return docs

    def submit_task(self, headers, mode=0) -> str:
        """
        提交文档解析任务
        :param headers: 鉴权信息
        :param mode: 段落拆分模式，默认为0
        0 - 返回文档的原始段落，不做其他处理
        1 - 根据标注的书签或目录分段，一般适合有层级标签的word文档
        2 - 根据内容里的章节条分段，适合制度类文档
        3 - 根据长度分段，默认按照500字拆分，会尽量保留完整句子
        :return:
        """
        if mode not in [0, 1, 2, 3]:
            raise Exception('Unsupported mode value.')
        payload = {'mode': mode}
        files = [
            ('file', (self.split_config.file_name, open(self.split_config.file_path, 'rb')))
        ]
        res = requests.post(self.split_config.upload_url(), headers=headers,
                            data=payload, files=files, proxies=self.proxies)
        if res.status_code == 200:
            res = res.json()
            if 'task_id' in res:
                return res['task_id']
        raise Exception(f'upload file failed, error code is {res["error_code"]}, reason is {res["error_msg"]}')

    def get_task_result(self, task_id, headers):
        res = requests.get(self.split_config.result_url(task_id), headers=headers, proxies=self.proxies).json()
        while res['task_status'] not in ['SUCCESS', 'ERROR']:
            time.sleep(3)
            res = requests.get(self.split_config.result_url(task_id), headers=headers, proxies=self.proxies).json()
        if res['task_status'] == 'SUCCESS':
            return res['result']
        raise Exception(f'task result failed, error code is {res["error_code"]}, reason is {res["error_msg"]}')
