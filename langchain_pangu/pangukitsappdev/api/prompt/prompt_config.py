#  Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
from typing import Optional

from pydantic.v1 import Field

from langchain_pangu.pangukitsappdev.api.config_loader import SdkBaseSettings


class PromptConfig(SdkBaseSettings):
    """Prompt参数
    Attributes:
        custom_prompt_path: 用户自定义prompt路径，优先生效
        default_prompt_path: 预制模板路径
    """
    custom_prompt_path: Optional[str] = Field(env="sdk.prompt.path")
    default_prompt_path: str = Field(default="prompts/default/")


class PromptTemplatesFileConfig(SdkBaseSettings):
    """模板文件存放结构
    Attributes:
        documents_stuff: documents/stuff.pt
        documents_refine_qa: documents/refine_qa.pt
        documents_refine_combine: documents/refine_combine.pt
        documents_mapreduce_map: documents/mapreduce_map.pt
        documents_mapreduce_reduce: documents/mapreduce_reduce.pt
        documents_summarize_stuff: documents/summarize_stuff.pt
        documents_summarize_refine_qa: documents/summarize_refine_qa.pt
        documents_summarize_refine_combine: documents/summarize_refine_combine.pt
        documents_summarize_mapreduce_map: documents/summarize_mapreduce_map.pt
        documents_summarize_mapreduce_reduce: documents/summarize_mapreduce_reduce.pt
        memory_summary: memory/summary.pt
        system_out_put_parser: system/out_put_parser.pt
        agent_react: agent/react.pt
        agent_react_pangu: agent/react_pangu.pt
        agent_tool_desc: agent/tool_desc.pt
        agent_tool_desc_pangu: agent/tool_desc_pangu.pt
        agent_tool_json_schema: agent/tool_json_schema.pt
        agent_react_pangu_2: agent/react_pangu_2.pt
        agent_tool_desc_pangu_2: agent/tool_desc_pangu_2.pt
        agent_react_pangu_unify: agent/react_pangu_unify.pt
        conversation_default: conversation/default.pt
        skill_conversation_rewrite: skill/conversation_rewrite.pt
        skill_agent_session_summary: skill/agent_session_summary.pt
        question_long_answer: question/long_answer.jinja2
        question_short_answer: question/short_answer.jinja2
        question_qa_with_sources_stuff: question/qa_with_sources_stuff.jinja2
        question_question_only: question/question_only.jinja2
    """
    documents_stuff: str = Field(default="documents/stuff.pt")
    documents_refine_qa: str = Field(default="documents/refine_qa.pt")
    documents_refine_combine: str = Field(default="documents/refine_combine.pt")
    documents_mapreduce_map: str = Field(default="documents/mapreduce_map.pt")
    documents_mapreduce_reduce: str = Field(default="documents/mapreduce_reduce.pt")
    documents_summarize_stuff: str = Field(default="documents/summarize_stuff.pt")
    documents_summarize_refine_qa: str = Field(default="documents/summarize_refine_qa.pt")
    documents_summarize_refine_combine: str = Field(default="documents/summarize_refine_combine.pt")
    documents_summarize_mapreduce_map: str = Field(default="documents/summarize_mapreduce_map.pt")
    documents_summarize_mapreduce_reduce: str = Field(default="documents/summarize_mapreduce_reduce.pt")
    memory_summary: str = Field(default="memory/summary.pt")
    system_out_put_parser: str = Field(default="system/out_put_parser.pt")
    agent_react: str = Field(default="agent/react.pt")
    agent_react_pangu: str = Field(default="agent/react_pangu.pt")
    agent_tool_desc: str = Field(default="agent/tool_desc.pt")
    agent_tool_desc_pangu: str = Field(default="agent/tool_desc_pangu.pt")
    agent_tool_json_schema: str = Field(default="agent/tool_json_schema.pt")
    agent_tool_simple_schema: str = Field(default="agent/tool_simple_schema.pt")
    agent_react_pangu_2: str = Field(default="agent/react_pangu_2.pt")
    agent_tool_desc_pangu_2: str = Field(default="agent/tool_desc_pangu_2.pt")
    agent_react_pangu_unify: str = Field(default="agent/react_pangu_unify.pt")
    conversation_default: str = Field(default="conversation/default.pt")
    skill_conversation_rewrite: str = Field(default="skill/conversation_rewrite.pt")
    skill_agent_session_summary: str = Field(default="skill/agent_session_summary.pt")
    question_long_answer: str = Field(default="question/long_answer.jinja2")
    question_short_answer: str = Field(default="question/short_answer.jinja2")
    question_qa_with_sources_stuff: str = Field(default="question/qa_with_sources_stuff.jinja2")
    question_question_only: str = Field(default="question/question_only.jinja2")
