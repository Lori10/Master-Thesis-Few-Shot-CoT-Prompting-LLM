from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFacePipeline, AzureOpenAI
import openai 
import load_env_vars
import os 

prompt_template='Q: {question}\nA: Let\'s think step by step.'
prompt = PromptTemplate(input_variables=["question"], template=prompt_template)

openai.api_key = os.getenv("OPENAI_API_KEY")
headers = {
            "x-api-key": openai.api_key,
        }

#gpt-3.5-turbo
llm = AzureOpenAI(
deployment_name='gpt-3.5-turbo',
model_name='gpt-3.5-turbo',
temperature=0,
headers=headers,
max_tokens=1024,
)

llm_chain = LLMChain(prompt=prompt, llm=llm, verbose=False)
response = llm_chain.run(question="Who is the president of the United States of America?")
print(response)