from constant_variables import COST_PER_TOKEN, ESTIMATE_COMPLETION_TOKENS, PREFIX_DIC, SUFFIX_DIC
from generate_fewshot_demonstrations import generate_severalcontexts_fewshot_random_demonstrations
from langchain.prompts import PromptTemplate
from tqdm import tqdm
import tiktoken

def num_tokens_from_string(string: str, model_name: str) -> int:
    """
        Given a string, return the number of tokens in the string using the tokenizer of a given LLM.
    """
    encoding = tiktoken.encoding_for_model(model_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def estimate_costs(seeds: list, dataset_name: str, dataset: list, strategy: str, model_name: str, nr_examples: int):
    """
        Estimate the costs for a given strategy and dataset.
    """

    total_costs = 0

    suffix_data = SUFFIX_DIC[dataset_name][strategy]
    if dataset_name == 'aqua':
        suffix_subset = suffix_data['subset']
        suffix_question = suffix_data['suffix']
    else:
        suffix = suffix_data
        
    prefix = PREFIX_DIC[dataset_name][strategy]

    list_contexts, _ = generate_severalcontexts_fewshot_random_demonstrations(seeds, dataset_name, dataset, nr_examples, strategy)
    
    if dataset_name == 'strategyqa' and len(list_contexts) == 1:
        seeds = ['no_seed']

    for context in tqdm(list_contexts):
        for example in dataset:
            # build the suffix
            if dataset_name == 'aqua':                
                formatted_suffix_subset = suffix_subset.format(example['options'])
                suffix = suffix_question + formatted_suffix_subset
            
            # build the template using prefix, context and suffix
            template = prefix + context + suffix
            prompt_template = PromptTemplate(input_variables=["question"], template=template)
            formatted_prompt = prompt_template.format(question=example['question'])
            example_total_tokens = num_tokens_from_string(formatted_prompt, model_name) + ESTIMATE_COMPLETION_TOKENS[strategy] 
            example_total_cost = COST_PER_TOKEN[model_name] * example_total_tokens
            total_costs += example_total_cost

    with open(f'../estimate_costs/{dataset_name}_{strategy}_lendata_{len(dataset)}.txt', 'w') as file:
        file.write(f'The estimated costs using the following parameters: dataset_name={dataset_name}, strategy={strategy}, nr_examples={nr_examples}, len_of_dataset={len(dataset)}, nr_runs={len(list_contexts)}, are {total_costs}')

    print('Estimate Costs Run Successfully')
    return total_costs