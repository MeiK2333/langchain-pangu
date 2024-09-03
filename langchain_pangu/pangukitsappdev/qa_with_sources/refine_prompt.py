#  Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.

from langchain.prompts import PromptTemplate

CN_DEFAULT_REFINE_PROMPT_TMPL = """
【新给出的上下文信息】：
------------
{context_str}
------------
【给定问题】：
{question}
【已有答案】：
{existing_answer}

请再结合【新给出的上下文信息】和【已有答案】，继续回答【给定问题】。
请严格按照如下要求进行回答：
1. 如果【新给出的上下文信息】没有帮助，则不做任何修改直接返回【已有答案】的内容。
2. 请直接返回最终答案，不要返回回答的过程。
3. 答案内容中删除“根据”关键字。
4. 如果根据给出的信息你无法回答，就说不知道，不要编造答案。
"""

CN_DEFAULT_REFINE_PROMPT = PromptTemplate(
    input_variables=["question", "existing_answer", "context_str"],
    template=CN_DEFAULT_REFINE_PROMPT_TMPL,
)

CN_DEFAULT_TEXT_QA_PROMPT_TMPL = (
    "上下文信息如下 \n"
    "---------------------\n"
    "{context_str}"
    "\n---------------------\n"
    "没有任何先验信息，请根据给出的上下文信息回答【给定问题】：{question}\n"
    "如果根据给出的信息你无法回答，就说不知道，不要编造答案。"
)

CN_DEFAULT_TEXT_QA_PROMPT = PromptTemplate(
    input_variables=["context_str", "question"], template=CN_DEFAULT_TEXT_QA_PROMPT_TMPL
)