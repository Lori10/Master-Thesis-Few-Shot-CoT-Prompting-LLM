import random
from extract_final_answer import extract_answer_gsm8k

# AQUA 
def generate_singlecontext_fewshot_random_demonstrations_aqua(selected_examples: list, 
                                                              strategy: str) -> str:
    """
        Given few-shot demonstrations from AQUA dataset and a strategy (standard or CoT), build a contact by concatenating the demonstrations.

        Args:
            selected_examples (list): a list of few-shot demonstrations
            strategy (str): strategy should be either standard or CoT

        Returns:
            context (str): a string which contains the few-shot demonstrations 
    """

    context = ''
    for prompt_example in selected_examples:
        context += f"Question: {prompt_example['question']}\n"
        context += f"Options: {prompt_example['options']}\n"

        if strategy == 'cot':
            context += f"Rationale: {prompt_example['rationale']}\n"
            
        context += f"Output: {prompt_example['correct']}\n\n"

    return context
                
# GSM8K 
def generate_singlecontext_fewshot_random_demonstrations_gsm8k(selected_examples: list ,
                                                               strategy: str) -> str:
    """
        Given few-shot demonstrations from GSM8K dataset and a strategy (standard or CoT), build a contact by concatenating the demonstrations.

        Args:
            selected_examples (list): a list of few-shot demonstrations
            strategy (str): strategy should be either standard or CoT

        Returns:
            context (str): a string which contains the few-shot demonstrations 
    """

    context = ''
    for prompt_example in selected_examples:
        context += f'Question: {prompt_example["question"]}\n'
        
        if strategy == 'standard':
            context += f'Output: {extract_answer_gsm8k(prompt_example["answer"])}\n\n'
        elif strategy == 'cot':
            context += f'Rationale: {prompt_example["answer"]}\n'
            context += f'Output: {extract_answer_gsm8k(prompt_example["answer"])}\n\n'
        else:
            print(f'Strategy must be "standard" OR "cot"')
            return None

    return context

# StrategyQA
def generate_singlecontext_fewshot_random_demonstrations_strategyqa(selected_examples: list, strategy: str) -> str:
    """
        Given few-shot demonstrations from StrategyQA dataset, build a contact by concatenating the demonstrations.

        Args:
            selected_examples (list): a list of few-shot demonstrations

        Returns:
            context (str): a string which contains the few-shot demonstrations 
    """

    if strategy == 'cot':
        with open("strategyqa_prompt_cot.txt", "r") as f:
            context = f.read()
    else:
        context = ''

        for prompt_example in selected_examples:
            context += f'Question: {prompt_example["question"]}\n'
            context += f"Output: {'yes' if prompt_example['answer'] else 'no'}\n\n"

    return context
    
# Generate several contexts
def generate_severalcontexts_fewshot_random_demonstrations(seeds: list, dataset_name: str, dataset: list, 
                                                           nr_examples: int, strategy: str) -> list:
    """
        Generate several contexts (few-shot demonstrations) using different seeds for a given
        strategy (standard or CoT) which are randomly sampled from a given dataset.

        Args:
            seeds (list): a list of seeds
            dataset_name (str): the name of the dataset
            dataset (list): a list of examples from the dataset
            nr_exampels (int): the number of generated demonstrations
            strategy (str): the strategy should be standard or CoT

        Returns:
            list_contexts (list): a list of contexts where each context contains randomly generated
                                  few-shot demonstrations
    """

    list_contexts = []
    examples_no_test = {}

    # For StandardQA and CoT we use the same promptinng demonstrations for all exampels in the dataset
    if strategy == 'cot' and dataset_name == 'strategyqa':
        with open("prompts/strategyqa_prompt_cot.txt", "r") as f:
            cot_context = f.read()
        list_contexts.append(cot_context)
        return list_contexts, None
    
    else:
        list_contexts = []

        for seed in seeds:    
            random.seed(seed)
            selected_examples = random.sample(dataset, nr_examples)
            examples_no_test[seed] = selected_examples

            if dataset_name == 'gsm8k':
                context = generate_singlecontext_fewshot_random_demonstrations_gsm8k(selected_examples, strategy)
                list_contexts.append(context)
            elif dataset_name == 'aqua':
                context = generate_singlecontext_fewshot_random_demonstrations_aqua(selected_examples, strategy)
                list_contexts.append(context)
            elif dataset_name == 'strategyqa':
                context = generate_singlecontext_fewshot_random_demonstrations_strategyqa(selected_examples, strategy)
                list_contexts.append(context)
            
        return list_contexts, examples_no_test