from langchain.llms import OpenAI
from paperqa_tool import PaperQATool
from langchain.agents import initialize_agent
from fire import Fire
import os
from copy import deepcopy
from langchain.agents import AgentType
import re
import pickle 

def answer(question: str, doc_store: str = "VectorStore/embeddings.pkl"):
    os.environ["BING_SUBSCRIPTION_KEY"] = "2075a41e8bd2447eae6595b3d4230ff8"
    os.environ["BING_SEARCH_URL"] = "https://api.bing.microsoft.com/v7.0/search"

    with open(doc_store, "rb") as f:
        docs = pickle.load(f)
    llm = OpenAI(temperature=0.1)
    tools = [PaperQATool(docs)]
    self_ask_with_search_agent = initialize_agent(tools, llm, agent=AgentType.SELF_ASK_WITH_SEARCH, verbose=True)

    my_agent = deepcopy(self_ask_with_search_agent)
    with open('prompts/strategyqa_prompt_cot_new.txt', 'r') as file:
        content = file.read()

    add_template = 'You are willing to answer questions that require reasoning. The final answer should be either "Yes" or "No". You must generate at least 2 follow up questions before generating the final answer. Follow the format of the examples below.\n'
    fewshot = content.replace('Question', 'question').replace('Answer', 'answer')
    preprocessed_fewshot = re.sub(r'Sub answer\s#\d+ :', 'Intermediate answer:', re.sub(r'Sub question\s#\d+ :', 'Follow up:', fewshot)).replace('question:', 'Question:').replace('Final answer:', 'So the final answer is:')
    output_parser = 'Question: {input}\nOutput:{agent_scratchpad}'
    my_agent.agent.llm_chain.prompt.template = add_template + preprocessed_fewshot + output_parser
    final_answer = my_agent.run(question)
    print(f'Final Answer: {final_answer}')
    final_answer_rating = int(input(f'Is the final answer reasonable? (1 to 5)'))
    #cot_explanation = my_agent.tools[0].api_wrapper.explanation

    ai_cot_explanation = []
    human_cot_explanation = []
    # for step in cot_explanation: 
    #     if isinstance(step, tuple):
    #         human_step = step[0]
    #         ai_step = step[1]
    #         human_cot_explanation.append(human_step)
    #         ai_cot_explanation.append(ai_step)
    #     else:
    #         human_cot_explanation.append(ai_step)
    #         ai_cot_explanation.append(ai_step)


    with open(doc_store, "wb") as f:
        pickle.dump(my_agent.tools[0].api_wrapper.docs, f)

    

    # human_row_cot = {'question': question, 'rationale': human_cot_explanation, 'answer': final_answer}
    # ai_row_cot = {'question': question, 'rationale': ai_cot_explanation, 'answer': final_answer}
    # cot_finetuning_positive_examples_file_path = '../human_feedback_cot_data/positive_data.jsonl'
    # if os.path.exists(cot_finetuning_positive_examples_file_path):
    #     mode = 'a'  # Append to existing file
    # else:
    #     mode = 'w'  # Create new file

    # with jsonlines.open(cot_finetuning_positive_examples_file_path, mode=mode) as writer:
    #     writer.write(human_row_cot)
    #     writer.write(ai_row_cot)

    # row = {'question': 'question3', 'rationale': 'cot_explanation3', 'answer': 'final_answer3'}
    # df = pd.DataFrame(row_cot, index=[0])
    # if os.path.exists(cot_finetuning_positive_examples_file_path):
    #     df.to_csv(cot_finetuning_positive_examples_file_path, mode='a', index=False, header=False)
    # else:
    #     df.to_csv(cot_finetuning_positive_examples_file_path, index=False)

    #cot_finetuning_negative_examples_file_path = '../human_feedback_cot_data/negative_data.csv'
if __name__ == '__main__':
    Fire(answer)

    