from langchain.llms import HuggingFacePipeline, AzureOpenAI 
from langchain.chat_models import AzureChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import openai 
import os 
import load_env_vars

prompt_template = "You are good at performing tasks that require reasoning abilities. Answer the following question.\nQ: {question}\nA: Let's think step by step."
model_id = 'gpt-35-turbo-0613'
#model_id = 'text-davinci-003'
prompt = PromptTemplate(input_variables=["question"], template=prompt_template)

openai.api_key = os.getenv("OPENAI_API_KEY")
headers = {
    "x-api-key": openai.api_key,
}

llm = AzureOpenAI(
deployment_name=model_id,
temperature=0.0,
headers=headers,
max_tokens=1024,
)

llm_chain = LLMChain(prompt=prompt, llm=llm, verbose=False)
response = llm_chain.run(question="Stefan goes to a restaurant to eat dinner with his family. They order an appetizer that costs $10 and 4 entrees that are $20 each. If they tip 20% of the total for the waiter, what is the total amount of money that they spend at the restaurant?")
print(response)