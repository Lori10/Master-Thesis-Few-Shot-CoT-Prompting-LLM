from experiment_llm import several_runs_same_context_for_run
from load_datasets import load_aqua, load_gsm8k, load_strategyqa
from fire import Fire
from constant_variables import PATH_AQUA_TEST, PATH_GSM8K_TEST, PATH_STRATEGYQA_TRAIN
import random 

def main(
        dataset_name: str = 'gsm8k', 
        strategy: str = 'standard', 
        seeds : list = [10, 20, 30], 
        model_name : str ='gpt-3.5-turbo', 
        nr_examples : int = 6, 
        dataset_size : int = None):
    
    if dataset_name == 'aqua':
        data = load_aqua(PATH_AQUA_TEST)    
    elif dataset_name == 'gsm8k':
        data = load_gsm8k(PATH_GSM8K_TEST)
    elif dataset_name == 'strategyqa':
        data = load_strategyqa(PATH_STRATEGYQA_TRAIN)
    else:
        print('Parameter dataset_name should be one of the following values "gsm8k", "aqua" or "strategyqa"')

    if dataset_size:
        random.seed(42)
        random.shuffle(data)
        data = data[:dataset_size]

    several_runs_same_context_for_run(seeds, dataset_name, data, strategy, model_name, nr_examples)

if __name__ == '__main__':
    Fire(main)



