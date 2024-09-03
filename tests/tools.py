import os
from typing import Literal
from typing import Type

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field

from langchain_pangu import ChatPanGu
from langchain_pangu.pangukitsappdev.api.llms.llm_config import (
    LLMConfig, LLMModuleConfig,
)
from langchain_pangu.pangukitsappdev.tool.tool import Tool

os.environ["SDK_CONFIG_PATH"] = "./llm.properties"


class Meeting(Tool):
    class MeetingParam(BaseModel):
        start: str = Field(description="会议开始时间，格式为yyyy-MM-dd HH:mm")
        end: str = Field(description="会议结束时间，格式为yyyy-MM-dd HH:mm")
        meetingRoom: str = Field(description="会议室")

    name = "reserve_meeting_room"
    description = "预订会议室"
    args_schema: Type[BaseModel] = MeetingParam
    principle = "请在需要预订会议室时调用此工具"
    input_desc = "会议开始与结束时间"
    output_desc = "会议预订的结果"

    def _run(self, start: str, end: str, meetingRoom: str) -> str:
        return "预订成功"


class Weather(Tool):
    class WeatherParam(BaseModel):
        city: str = Field(description="要查询的城市")

    name = "weather"
    description = "查询天气"
    args_schema: Type[BaseModel] = WeatherParam
    principle = "请在需要查询城市天气时调用此工具"
    input_desc = "要查询的城市"
    output_desc = "城市的天气"

    def _run(self, city: str) -> str:
        print("查天气")
        return "下刀子雨"


tools = [Meeting(), Weather()]

tool_node = ToolNode(tools)

model = ChatPanGu(
    llm_config=LLMConfig(
        llm_module_config=LLMModuleConfig(module_version="N2_agent_v2")
    )
).bind_tools(tools)


# Define the function that determines whether to continue or not
def should_continue(state: MessagesState) -> Literal['tools', END]:
    messages = state['messages']
    last_message = messages[-1]
    # If the LLM makes a tool call, then we route to the "tools" node
    if last_message.tool_calls:
        return "tools"
    # Otherwise, we stop (reply to the user)
    return END


# Define the function that calls the model
def call_model(state: MessagesState):
    messages = state['messages']
    response = model.invoke(messages)
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}


# Define a new graph
workflow = StateGraph(MessagesState)

# Define the two nodes we will cycle between
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)

# Set the entrypoint as `agent`
# This means that this node is the first one called
workflow.add_edge(START, "agent")

# We now add a conditional edge
workflow.add_conditional_edges(
    # First, we define the start node. We use `agent`.
    # This means these are the edges taken after the `agent` node is called.
    "agent",
    # Next, we pass in the function that will determine which node is called next.
    should_continue,
)

# We now add a normal edge from `tools` to `agent`.
# This means that after `tools` is called, `agent` node is called next.
workflow.add_edge("tools", 'agent')

# Initialize memory to persist state between graph runs
checkpointer = MemorySaver()

# Finally, we compile it!
# This compiles it into a LangChain Runnable,
# meaning you can use it as you would any other runnable.
# Note that we're (optionally) passing the memory when compiling the graph
app = workflow.compile(checkpointer=checkpointer)

# Use the Runnable
final_state = app.invoke(
    {"messages": [HumanMessage(content="北京天气如何？")]},
    config={"configurable": {"thread_id": 42}}
)
print(final_state["messages"][-1].content)  # 北京现在下刀子雨，请带好雨具。
