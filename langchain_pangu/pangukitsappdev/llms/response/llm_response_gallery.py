#  Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
from langchain_pangu.pangukitsappdev.api.schema import LLMResp
from langchain_pangu.pangukitsappdev.llms.response.gallery_text_resp import GalleryTextResp


class LLMRespGallery(LLMResp):
    """gallery response封装结构体
    Attributes:
        gallery_text_resp: api返回response结构体
    """
    gallery_text_resp: GalleryTextResp
