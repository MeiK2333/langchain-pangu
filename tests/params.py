from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser

from langchain_pangu import ChatPanGu

parser = StrOutputParser()
model = ChatPanGu(
    pangu_url="",
    # ak="",
    # sk="",
    domain="",
    user="",
    password="",
    iam_url="",
    model_version="N2_agent",
    project="",
)

messages = [HumanMessage(content="你好")]


resp = model.invoke(messages)
print(resp)
