from experiment_llm import single_run_different_contexts
from load_datasets import load_aqua, load_gsm8k, load_strategyqa
from fire import Fire
from constant_variables import PATH_AQUA_TEST, PATH_GSM8K_TEST, PATH_STRATEGYQA_TRAIN
from save_load_results import save_results_different_contexts
import random 

def main(
        dataset_name: str = 'gsm8k', 
        strategy: str = 'standard', 
        nr_runs: int = 2, 
        seed: int = 42,
        model_name : str ='gpt-3.5-turbo', 
        nr_examples : int = 2, 
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

    seeds_per_run = []
    random.seed(seed)
    for _ in range(nr_runs):
        seeds = [random.randint(0, 5000) for _ in range(len(data))]
        seeds_per_run.append(seeds)

    df_list = []
    if dataset_name == 'strategyqa' and strategy == 'cot':
        result_df = single_run_different_contexts(seeds, dataset_name, data, strategy, model_name, nr_examples)
        df_list.append(result_df)
    else:
        for seeds in seeds_per_run:
            result_df = single_run_different_contexts(seeds, dataset_name, data, strategy, model_name, nr_examples)
            df_list.append(result_df)
    
    save_results_different_contexts(df_list, dataset_name, strategy, seed)
if __name__ == '__main__':
    Fire(main)



