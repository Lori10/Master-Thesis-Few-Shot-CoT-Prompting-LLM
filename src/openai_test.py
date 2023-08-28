from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import AIMessage, HumanMessage, SystemMessage

llm = ChatOpenAI(openai_api_key='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', model='gpt-3.5-turbo-0613')

messages = [
    SystemMessage(
        content="You are a helpful assistant that translates English to French."
    ),
    HumanMessage(
        content="Translate this sentence from English to French. I love programming."
    ),
]
print(llm(messages))