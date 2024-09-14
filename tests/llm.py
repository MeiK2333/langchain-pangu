import asyncio

from langchain_core.prompts import ChatPromptTemplate

from langchain_pangu import PanGuLLM

llm = PanGuLLM(profile_file="./llm.properties")


async def main():
    prompt = ChatPromptTemplate.from_messages(
        [("system", "你是个机器人"), ("human", "{input}")]
    )
    chain = prompt | llm
    async for event in chain.astream_events({"input": "你好，你是谁？"}, version="v1"):
        print(event)


asyncio.run(main())


def main2():
    prompt = ChatPromptTemplate.from_messages(
        [("system", "你是个机器人"), ("human", "{input}")]
    )
    chain = prompt | llm
    for event in chain.stream({"input": "你好，你是谁？"}):
        print(event)


main2()
