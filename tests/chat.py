from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser

from langchain_pangu import ChatPanGu

parser = StrOutputParser()
model = ChatPanGu(profile_file="./llm.properties")

messages = [HumanMessage(content="你好")]

resp = model.invoke(messages)
print(resp)
chain = model | parser
print(chain.invoke(messages))
