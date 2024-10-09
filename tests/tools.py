import os
from typing import Literal

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode

from langchain_pangu import ChatPanGu
from langchain_pangu.pangukitsappdev.api.llms.llm_config import (
    LLMConfig,
    LLMModuleConfig,
)

os.environ["SDK_CONFIG_PATH"] = "./llm.properties"


@tool
def search(city: str):
    """
    查询指定城市天气
    """
    return f"{city} 今天在下刀子"


@tool
def compare(a: float, b: float):
    """
    比较两个数字大小
    """
    print(f"调用工具比对数字：{a} 和 {b}")
    if a > b:
        return f"{a} 更大"
    elif b > a:
        return f"{b} 更大"
    return f"{a} 和 {b} 一样大"


tools = [search, compare]

tool_node = ToolNode(tools)

model = ChatPanGu(profile_file="./llm.properties").bind_tools(tools)


# Define the function that determines whether to continue or not
def should_continue(state: MessagesState) -> Literal["tools", END]:
    messages = state["messages"]
    last_message = messages[-1]
    # If the LLM makes a tool call, then we route to the "tools" node
    if last_message.tool_calls:
        return "tools"
    # Otherwise, we stop (reply to the user)
    return END


# Define the function that calls the model
def call_model(state: MessagesState):
    messages = state["messages"]
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
workflow.add_edge("tools", "agent")

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
    # {"messages": [HumanMessage(content="13.11 和 13.8 哪个大？")]},
    config={"configurable": {"thread_id": 42}},
)
print(final_state["messages"][-1].content)  # 北京现在下刀子雨，请带好雨具。
