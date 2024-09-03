#  Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
from langchain.prompts import PromptTemplate

template = """请仔细阅读以下文章回答问题，不要遗漏任何信息，尽可能的覆盖全面，给出全面详细的回答。
如果问题与文章不相关，则无需参考文章内容，直接回答“抱歉我没有找到相关的参考资料。
文章:
=========
{% for item in documents %}{{ item.pageContent }}{% if not loop.last %},{% endif %}{% endfor %}
=========

问题：{{question}}
回答：
"""

input_variables = ["summaries", "question"]
PROMPT = PromptTemplate.from_template(template, template_format="jinja2")
