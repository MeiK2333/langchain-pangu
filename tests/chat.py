import os

from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser

from langchain_pangu import ChatPanGu
from langchain_pangu.pangukitsappdev.api.llms.llm_config import LLMConfig

os.environ["SDK_CONFIG_PATH"] = "./llm.properties"
parser = StrOutputParser()
model = ChatPanGu(llm_config=LLMConfig())

messages = [
    HumanMessage(content='你好')
]

resp = model.invoke(messages)
print(resp)
chain = model | parser
print(chain.invoke(messages))
