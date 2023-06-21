from copy import deepcopy
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.llms import OpenAI
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from retrievalqa_human_tool import HumanQATool
import re
from fire import Fire
from constant_variables import INDEX_PATH

def answer(query: str):
    db = FAISS.load_local(INDEX_PATH, OpenAIEmbeddings())
    llm = OpenAI(temperature=0) #, model_name='gpt-3.5-turbo')
    tools = [HumanQATool(db)]
    self_ask_with_search_agent = initialize_agent(tools, llm, agent=AgentType.SELF_ASK_WITH_SEARCH, verbose=True)
    my_agent = deepcopy(self_ask_with_search_agent)
    with open('prompts/strategyqa_prompt_cot_new.txt', 'r') as file:
        content = file.read()

    add_template = 'You are willing to answer questions that require reasoning. The final answer should be either "Yes" or "No". You must generate at least 2 follow up questions before generating the final answer. Let"s think step by step! \n'
    fewshot = content.replace('Question', 'question').replace('Answer', 'answer')
    preprocessed_fewshot = re.sub(r'Sub answer\s#\d+ :', 'Intermediate answer:', re.sub(r'Sub question\s#\d+ :', 'Follow up:', fewshot)).replace('question:', 'Question:').replace('Final answer:', 'So the final answer is:')
    output_parser = 'Question: {input}\nOutput:{agent_scratchpad}'
    my_agent.agent.llm_chain.prompt.template = add_template + preprocessed_fewshot + output_parser
    answer = my_agent.run(query)
    print(answer)
    
if __name__ == '__main__':
    Fire(answer)
