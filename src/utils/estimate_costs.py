import tiktoken
from utils.embedding_generation import initialize_embedding_model

COST_PER_TOKEN = {'text-davinci-003' : 0.02/1000, 
                  'gpt-35-turbo-0613' : 0.0015/0.002,
                  'text-embedding-ada-002' : 0.0004/1000,
                  'text-embedding-ada-002-v2' : 0.0004/1000}

def num_tokens_from_string(string: str, model_name: str) -> int:
    """
        Given a string, return the number of tokens in the string using the tokenizer of a given LLM.
    """
    encoding = tiktoken.encoding_for_model(model_name)
    #encoding = tiktoken.get_encoding(model_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def single_run_costs(data_loader, args, prompt_template):
    costs = 0
    
    for example in data_loader:
        formatted_prompt = prompt_template.format(question=example['question'])
        example_total_tokens = num_tokens_from_string(formatted_prompt, args.model_id) + 100 
        costs += COST_PER_TOKEN[args.model_id] * example_total_tokens
    return costs

def all_prompts_costs(args, data_loader, prompts_list):
    all_costs = []
    for i in range(len(prompts_list)):
        cost_prompt = single_run_costs(data_loader, args, prompts_list[i])
        all_costs.append(cost_prompt)
        
    return all_costs

def uncertainty_cost_all_examples(args, dataloader):
    all_costs = 0
    for example in dataloader:
        single_example_cost = uncertainty_cost_single_example(args, example)
        all_costs += single_example_cost

    return all_costs

def uncertainty_cost_single_example(args, example):
    formatted_prompt = args.prompt_template.format(question=example['question'])
    example_total_tokens = num_tokens_from_string(formatted_prompt, args.model_id) + 100 
    return COST_PER_TOKEN[args.model_id] * example_total_tokens * args.num_trails


def embedding_cost(args, dataloader):
    costs = 0
    for example in dataloader:
        costs += COST_PER_TOKEN[args.embedding_model_id] * num_tokens_from_string(example['question'], args.embedding_model_id) 
    return costs