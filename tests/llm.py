import os

from langchain_core.prompts import ChatPromptTemplate

from langchain_pangu import PanGuLLM
from langchain_pangu.pangukitsappdev.api.llms.llm_config import LLMConfig

os.environ["SDK_CONFIG_PATH"] = "./llm.properties"
llm = PanGuLLM(
    llm_config=LLMConfig(),
)

import asyncio


async def main():
    prompt = ChatPromptTemplate.from_messages(
        [("system", "你是个机器人"), ("human", "{input}")]
    )
    chain = prompt | llm
    async for event in chain.astream_events({"input": "你好，你是谁？"}, version="v1"):
        print(event)


asyncio.run(main())
