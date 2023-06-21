import os 
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from index_strategyqa_retrievalqa import QASTool
from langchain.llms import OpenAI
from fire import Fire 
from index_strategyqa_retrievalqa import load_index
import copy

def answer(query: str):

    os.environ["OPENAI_API_KEY"] = "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"

    db = load_index()
    llm = OpenAI(temperature=0.2)
    tools = [QASTool(db)]
    my_agent = initialize_agent(tools, llm, agent=AgentType.SELF_ASK_WITH_SEARCH, verbose=True,
                                max_iterations=1)
    
    # with open("../Prompts/strategyqa_prompt_cot.txt", "r") as f:
    #     context = f.read()

    context = """
    [Example 1]
    Question: Do hamsters provide food for any animals?
    Output:
    Sub Question #0 : What type of animals are hamsters?
    Sub Answer #0 : Hamsters are prey animals.
    Sub Question #1 : Can prey animals be food for other animals?
    Sub Answer #1 : Prey are food for predators.
    Sub Question #2 : Do hamsters provide food for any animals?
    Sub Answer #2 : Since hamsters are prey animals, and prey are food for predetors, hamsters provide food for some animals.
    Final Answer: YES
    """

    new_template = context + '\nQuestion: {input}\nOutput: {agent_scratchpad}\n'
    new_agent = copy.deepcopy(my_agent)
    new_agent.agent.llm_chain.prompt.template = new_template

    #print(my_agent.agent.llm_chain.prompt.template)
    #print('----------------------------------------------------------------')
    print(new_agent.agent.llm_chain.prompt.template)
    completion = new_agent.run(query)


if __name__ == '__main__':
    Fire(answer)