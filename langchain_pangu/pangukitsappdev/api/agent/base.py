#  Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
from __future__ import unicode_literals

import json
import logging
import uuid
from abc import ABC, abstractmethod
from json import JSONDecodeError
from typing import List, Union, Optional, Dict, Any

from langchain.schema.messages import AIMessageChunk, BaseMessage
from langchain.schema.output import ChatGenerationChunk, LLMResult
from pydantic.v1.json import pydantic_encoder

from langchain_pangu.pangukitsappdev.agent.agent_action import AgentAction
from langchain_pangu.pangukitsappdev.agent.agent_session import AgentSession
from langchain_pangu.pangukitsappdev.api.llms.base import LLMApi, ConversationMessage, Role
from langchain_pangu.pangukitsappdev.api.llms.llm_config import LLMModuleConfig
from langchain_pangu.pangukitsappdev.api.retriever.base import ToolRetriever
from langchain_pangu.pangukitsappdev.api.tool.base import AbstractTool
from langchain_pangu.pangukitsappdev.callback.StreamCallbackHandler import StreamCallbackHandler

logger = logging.getLogger(__name__)


class AgentListener(ABC):
    """Agent监听，允许对Agent的各个阶段进行处理
    """

    def on_session_start(self, agent_session: AgentSession):
        """
        Session启动时调用
        :param agent_session: AgentSession
        """

    def on_session_iteration(self, agent_session: AgentSession):
        """
        Session迭代过程中调用
        :param agent_session: AgentSession
        """

    def on_session_end(self, agent_session: AgentSession):
        """
        Session结束时调用
        :param agent_session: AgentSession
        """

    def on_check_interrupt_requirements(self, agent_session: AgentSession):
        """
        onSessionIteration调用结束后，检查Agent是否需要终止，如果需要终止，则返回true，默认不终止
        可以在终止前对agentSession进行修改，如：修改agent的finalAnswer
        :param agent_session: AgentSession
        :return: bool类型结果
        """
        return False


class Agent(ABC):

    @abstractmethod
    def add_tool(self, tool: AbstractTool):
        """
        为Agent增加工具类
        :param tool: Tool
        """

    @abstractmethod
    def remove_tool(self, tool_id: str):
        """
        删除tool
        :param tool_id: 工具id
        """

    @abstractmethod
    def clear_tool(self):
        """
        删除所有工具
        """

    @abstractmethod
    def run_step(self, agent_session: AgentSession) -> AgentSession:
        """
        单步执行Agent，提供干预能力
        :param agent_session: 初始状态或运行中状态的agentSession，包括历史消息及其执行步骤，可以使用AgentSessionHelper类辅助处理
        :return: Agent执行的结果
        """

    @abstractmethod
    def run(self, user_input: Union[List[ConversationMessage], str, AgentSession]) -> AgentSession:
        """
        递归执行Agent，直到完成所有步骤的执行
        :param user_input: 用户的输入, 支持多轮List[ConversationMessage]或str或AgentSession
        :return: 计划的结果
        """

    @abstractmethod
    def set_max_iterations(self, iterations: int):
        """
        设置最大迭代次数
        :param iterations: 次数
        """

    @abstractmethod
    def set_tool_retriever(self, tool_retriever: ToolRetriever):
        """
        设置工具检索器
        :param tool_retriever: 工具检索器
        """

    @abstractmethod
    def add_listener(self, agent_listener: AgentListener):
        """
        添加一个Agent监听器
        :param agent_listener: Agent监听器
        """

    @abstractmethod
    def set_stream_callback(self, text_stream_callback: StreamCallbackHandler,
                            tool_stream_callback: StreamCallbackHandler):
        """
        设置流式输出回调函数
        :param text_stream_callback: Agent文本输出的StreamCallback
        :param tool_stream_callback: Agent工具输出的StreamCallBack
        """


class AbstractAgent(Agent):

    def __init__(self, llm: LLMApi):
        """
        构造一个agent
        :param llm: LLMApi
        """
        self.llm = llm
        self.tool_map: Dict[str, AbstractTool] = {}
        self.max_iterations = 15
        self.agent_listener: Optional[AgentListener] = None
        self.tool_retriever: Optional[ToolRetriever] = None

    @staticmethod
    def convert_message_to_dict(conversation_messages: List[ConversationMessage]) -> List[Dict]:
        # 用于将python命名风格变量适配小驼峰，与Java版本保持prompt模板统一
        return [
            {
                "role": {
                    "text": message.role.text,
                    "desc": message.role.desc
                },
                "actions": [
                    {
                        "req": action.req,
                        "resp": action.resp,
                        "thought": action.thought,
                        "actionJson": action.action_json,
                        "action": action.action,
                        "actionInput": action.action_input,
                        "observation": action.observation,
                        "userFeedBack": action.user_feedback
                    } for action in message.actions
                ],
                "content": message.content
            } for message in conversation_messages]

    def add_tool(self, tool: AbstractTool):
        tool_id = tool.get_tool_id()
        self.tool_map.update({tool_id: tool})

    def remove_tool(self, tool_id: str):
        self.tool_map.pop(tool_id)

    def clear_tool(self):
        self.tool_map.clear()

    def run_step(self, agent_session: AgentSession) -> AgentSession:
        # tool检索器处理
        self.init_tool_from_retriever(agent_session.messages)
        # 关闭自动添加新的SystemMessage
        self.llm.get_llm_config().llm_module_config.enable_append_system_message = False
        agent_session.is_by_step = True
        # react 单步执行
        try:
            self.react(agent_session)
        except (ValueError, JSONDecodeError, TypeError) as e:
            logger.debug("run error when call react", e)
            raise e

        if agent_session.agent_session_status != "FINISHED":
            self.notice_session_iteration(agent_session, agent_session.current_action)
        return agent_session

    def run(self, user_input: Union[List[ConversationMessage], str, AgentSession]) -> AgentSession:
        """
        执行agent
        :param user_input: 用户的输入, 支持多轮List[ConversationMessage]或str或AgentSession
        :return: 计划的结果
        """
        agent_session = AgentSessionHelper.init_agent_session(user_input) if \
            isinstance(user_input, str) or isinstance(user_input, List) else user_input
        # tool检索器处理
        self.init_tool_from_retriever(agent_session.messages)
        # 关闭自动添加新的SystemMessage
        self.llm.get_llm_config().llm_module_config.enable_append_system_message = False
        self.notice_session_start(agent_session)
        agent_session.is_by_step = False
        # 新增一个Assistant回复
        if agent_session.messages[-1].role.text != "assistant":
            agent_session.current_message = ConversationMessage(role=Role.ASSISTANT, content="")
            agent_session.messages.append(agent_session.current_message)
        try:
            self.react(agent_session)
        except (ValueError, JSONDecodeError, TypeError) as e:
            logger.debug("run error when call react", e)
            raise e

        # 设置Assistant消息内容
        AgentSessionHelper.update_assistant_message(agent_session, False)
        actions = agent_session.current_message.actions
        if actions:
            logger.info(AgentSessionHelper.print_plan(agent_session))
        return agent_session

    @abstractmethod
    def react(self, agent_session: AgentSession):
        """
        迭代解决问题
        :param agent_session: 历史迭代几率
        """

    def add_listener(self, agent_listener: AgentListener):
        self.agent_listener = agent_listener

    def set_max_iterations(self, iterations: int):
        if iterations <= 0:
            raise ValueError("iterations value not legal.")
        self.max_iterations = iterations

    def set_tool_retriever(self, tool_retriever: ToolRetriever):
        self.tool_retriever = tool_retriever

    def notice_session_start(self, agent_session: AgentSession):
        agent_session.agent_session_status = "RUNNING"
        if self.agent_listener:
            self.agent_listener.on_session_start(agent_session)

    def notice_session_iteration(self, agent_session: AgentSession, action: AgentAction):
        agent_session.current_message.actions.append(action)
        agent_session.agent_session_status = "RUNNING"

        if self.agent_listener:
            self.agent_listener.on_session_iteration(agent_session)

    def notice_session_end(self, agent_session: AgentSession, action: AgentAction):
        agent_session.current_message.actions.append(action)
        agent_session.agent_session_status = "FINISHED"
        if self.agent_listener:
            self.agent_listener.on_session_end(agent_session)

    def tool_execute(self, tool: AbstractTool, tool_input: Union[str, dict], agent_session: AgentSession):
        if agent_session.is_by_step:
            return
        try:
            tool_result = tool.run(tool_input)
        except TypeError:
            tool_result = tool.run(str(tool_input))
        action = agent_session.current_action
        if isinstance(tool_result, (str, int, float, bool)):
            action.observation = str(tool_result)
        else:
            action.observation = json.dumps(tool_result, default=pydantic_encoder, ensure_ascii=False)
        # 本次迭代结束
        self.notice_session_iteration(agent_session, action)
        # 执行下一迭代
        self.react(agent_session)

    @staticmethod
    def sub_str_between(origin_str: str, start_str: str, end_str: str):
        if origin_str:
            start_pos = origin_str.find(start_str)
            if start_pos != -1:
                end_pos = origin_str.find(end_str)
                if end_pos != -1:
                    return origin_str[start_pos + len(start_str): end_pos]
        return ""

    @staticmethod
    def sub_str_before(origin_str: str, separator: str):
        if origin_str:
            if not separator:
                return ""
            else:
                pos = origin_str.find(separator)
                return origin_str if pos == -1 else origin_str[:pos]
        else:
            return origin_str

    @staticmethod
    def sub_str_after(origin_str: str, separator: str):
        if origin_str:
            if separator == "":
                return ""
            else:
                pos = origin_str.find(separator)
                return "" if pos == -1 else origin_str[pos + len(separator):]
        else:
            return origin_str

    @staticmethod
    def remove_start(origin_str: str, remove: str):
        if origin_str and remove:
            return origin_str[len(remove):] if origin_str.startswith(remove) else origin_str
        else:
            return origin_str

    def need_interrupt(self, agent_session: AgentSession) -> bool:
        if not agent_session.current_message.actions:
            return False
        # 超过最大迭代次数终止
        if len(agent_session.current_message.actions) >= self.max_iterations:
            logger.debug("stopped due to iteration limit. maxIterations is %s", self.max_iterations)
            return True
        # 用户终止
        if self.agent_listener and self.agent_listener.on_check_interrupt_requirements(agent_session):
            agent_session.agent_session_status = "INTERRUPTED"
            logger.info("agent stopped due to manual interruption")
            return True
        return False

    def init_tool_from_retriever(self, messages: List[ConversationMessage]):
        if self.tool_retriever is None or messages is None:
            return
        # 处理多轮对话
        preprocessor = self.tool_retriever.get_query_preprocessor()
        query = preprocessor(messages)
        tools = self.tool_retriever.search(query)
        self.tool_map.clear()
        for tool in tools:
            self.add_tool(tool)

    def get_system_prompt(self, agent_session: AgentSession):
        # 优先取设置到的LLM人设
        system_prompt = self.llm.get_llm_config().llm_module_config.system_prompt
        if system_prompt is not None:
            return system_prompt
        # 其次取message中最后一个人设
        for message in reversed(agent_session.messages):
            if message.role.text == "system":
                return message.content
        return None

    def set_stream_callback(self, text_stream_callback: StreamCallbackHandler,
                            tool_stream_callback: StreamCallbackHandler):
        self.llm.set_callback(AgentStreamCallBack(text_stream_callback=text_stream_callback,
                                                  tool_stream_callback=tool_stream_callback,
                                                  llm_module_config=self.llm.get_llm_config().llm_module_config))

    def get_tool(self, tool_id: str):
        if not self.tool_map:
            logger.error("there is no tool in agent")
            raise ValueError("there is no tool in agent")
        tool = self.tool_map.get(tool_id)
        if not tool:
            # 如果没有合适tool，则使用最相似的tool
            tool_map_keys = list(self.tool_map.keys())
            distance_list = list(map(AbstractAgent.levenshtein_distance,
                                     [tool_id for _ in range(len(tool_map_keys))],
                                     tool_map_keys))
            new_tool_id = tool_map_keys[distance_list.index(min(distance_list))]
            logger.warning("can not find tool for %s in %s, the most similar tool is %s",
                           tool_id, tool_map_keys, new_tool_id)
            return self.tool_map.get(new_tool_id)
        return tool

    @staticmethod
    def levenshtein_distance(tool_id: str, tool_map_key: str) -> int:
        if not tool_id or not tool_map_key:
            raise ValueError("tool_id must not be empty or None")
        if len(tool_id) < len(tool_map_key):
            left = tool_id
            right = tool_map_key
        else:
            left = tool_map_key
            right = tool_id
        left_len = len(left)
        right_len = len(right)
        # dp数组初始化
        dp = [[right_len for _ in range(right_len + 1)] for _ in range(left_len + 1)]
        for i in range(left_len + 1):
            dp[i][0] = i
        for i in range(right_len + 1):
            dp[0][i] = i
        for i in range(1, left_len + 1):
            for j in range(1, right_len + 1):
                cost = 0 if left[i - 1] == right[j - 1] else 1
                dp[i][j] = min(min(dp[i - 1][j] + 1, dp[i][j - 1] + 1), dp[i - 1][j - 1] + cost)
        return dp[left_len][right_len]


class AgentStreamCallBack(StreamCallbackHandler):
    def __init__(self, text_stream_callback: StreamCallbackHandler,
                 tool_stream_callback: StreamCallbackHandler,
                 llm_module_config: LLMModuleConfig):
        super().__init__()
        self.text_stream_callback = text_stream_callback
        self.tool_stream_callback = tool_stream_callback
        self.llm_module_config = llm_module_config
        self.serialized = {}
        self.messages = []
        self.in_bool_stream = False
        self.tool_generation: Optional[ChatGenerationChunk] = None

    def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[BaseMessage]],
        **kwargs: Any,
    ) -> Any:
        self.text_stream_callback.on_chat_model_start(serialized=serialized, messages=messages, **kwargs)
        self.serialized = serialized
        self.messages = messages

    def on_llm_error(
        self,
        error: BaseException,
        **kwargs: Any,
    ) -> Any:
        self.text_stream_callback.on_llm_error(error=error, **kwargs)
        self.tool_stream_callback.on_llm_error(error=error, **kwargs)

    def on_llm_end(
        self,
        response: LLMResult,
        **kwargs: Any,
    ) -> Any:
        self.text_stream_callback.on_llm_end(response=response, **kwargs)

    def on_llm_new_token(
        self,
        token: str,
        **kwargs: Any,
    ) -> Any:
        unify_tool_tag_prefix = self.llm_module_config.llm_module_property.unify_tool_tag_prefix
        unify_tool_tag_suffix = self.llm_module_config.llm_module_property.unify_tool_tag_suffix
        if unify_tool_tag_prefix in token:
            # 工具调用之前部分使用文本流输出
            text_token = AbstractAgent.sub_str_before(token, unify_tool_tag_prefix)
            self.text_stream_callback.on_llm_new_token(token=text_token, **kwargs)
            # 工具调用开始，工具调用后部分使用工具流输出
            self.tool_stream_callback.on_chat_model_start(serialized=self.serialized, messages=self.messages, **kwargs)
            self.in_bool_stream = True
            tool_token = AbstractAgent.sub_str_after(token, unify_tool_tag_prefix)
            self.tool_stream_callback.on_llm_new_token(token=tool_token, **kwargs)
            self.tool_generation = ChatGenerationChunk(message=AIMessageChunk(content=tool_token))
        elif unify_tool_tag_suffix in token:
            # 工具调用结束之前部分仍用工具流输出
            tool_token = AbstractAgent.sub_str_before(token, unify_tool_tag_suffix)
            self.tool_stream_callback.on_llm_new_token(token=tool_token, **kwargs)
            self.tool_generation += ChatGenerationChunk(message=AIMessageChunk(content=tool_token))
            # 工具调用结束
            self.tool_stream_callback.on_llm_end(response=LLMResult(generations=[[self.tool_generation]]))
            self.in_bool_stream = False
            # 工具调用后部分文本流输出
            text_token = AbstractAgent.sub_str_after(token, unify_tool_tag_suffix)
            self.text_stream_callback.on_llm_new_token(token=text_token, **kwargs)
        else:
            if self.in_bool_stream:
                # 工具流
                self.tool_stream_callback.on_llm_new_token(token=token, **kwargs)
                self.tool_generation += ChatGenerationChunk(message=AIMessageChunk(content=token))
            else:
                # 文本流
                self.text_stream_callback.on_llm_new_token(token=token, **kwargs)


class AgentSessionHelper:
    """
    AgentSession辅助类
    """
    FINAL_ACTION = "FINAL_ANSWER"

    @staticmethod
    def init_agent_session(user_message: Union[str, List[ConversationMessage]]) -> AgentSession:
        """
        使用用户消息初始化一个AgentSession
        :param user_message: 用户消息
        :return: agent_session
        """
        messages = [ConversationMessage(role=Role.USER,
                                        content=user_message)] if isinstance(user_message, str) else user_message
        agent_session = AgentSession(messages=messages, session_id=str(uuid.uuid4()),
                                     history_action=[], agent_session_status="INIT")
        # 新增一个Assistant回复
        agent_session.current_message = ConversationMessage(role=Role.ASSISTANT, content="")
        agent_session.messages.append(agent_session.current_message)
        return agent_session

    @staticmethod
    def update_assistant_message(agent_session: AgentSession, override: bool):
        """
        使用当前的action更新AssistantMessage的内容
        :param agent_session: AgentSession
        :param override: 是否覆盖已有的内容
        """
        if override or not agent_session.current_message.content:
            agent_session.current_message.content = str(agent_session.current_action.action_input)


    @staticmethod
    def set_tool_output(agent_session: AgentSession, observation: str):
        """
        向Agent的当前步骤设置工具返回结果
        :param agent_session: AgentSession
        :param observation: 工具返回结果
        """
        agent_session.current_action.observation = observation

    @staticmethod
    def set_user_feedback(agent_session: AgentSession, user_feedback: str):
        """
        向Agent的当前步骤设置用户反馈
        :param agent_session: AgentSession
        :param user_feedback: 用户反馈
        """
        agent_session.current_action.user_feedback = user_feedback

    @staticmethod
    def print_plan(agent_session: AgentSession) -> str:
        """
        打印AgentSession
        :param agent_session: AgentSession
        :return: AgentSession的字符串输出
        """
        log_msg = ""
        for message in agent_session.messages:
            log_msg += f"\n{message.role.desc}: {message.content}"
            # 如果为Assistant消息，打印可能存在的Action
            if message.role.text == "assistant" and message.actions:
                actions = message.actions
                for i in range(len(actions)):
                    log_msg += f"\n - 步骤{i+1}"
                    action = actions[i]
                    if action.thought and action.thought != action.action_input:
                        thought = action.thought.replace("\n", "")
                        log_msg += f":\n   思考:{thought}"
                    if action.action == "FINAL_ANSWER":
                        log_msg += f"\n   答复:{action.action_input}"
                    else:
                        log_msg += f"\n   行动:使用工具[{action.action}],传入参数{action.action_input}"
                        if action.observation:
                            log_msg += f"\n   工具返回:{action.observation}"
                        if action.user_feedback:
                            log_msg += f"\n   用户反馈:{action.user_feedback}"
        return log_msg
