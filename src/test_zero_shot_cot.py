from fire import Fire
from zero_shot_cot import zero_shot_cot
from constant_variables import PATH_GSM8K_TEST, PATH_STRATEGYQA_TRAIN, PATH_AQUA_TEST
from load_datasets import load_gsm8k, load_strategyqa, load_aqua
import random 

def main(dataset_name: str = 'strategyqa', dataset_size: int = None):
    if dataset_name == 'gsm8k':
        data = load_gsm8k(PATH_GSM8K_TEST)
    elif dataset_name == 'aqua':
        data = load_aqua(PATH_AQUA_TEST)
    elif dataset_name == 'strategyqa':
        data = load_strategyqa(PATH_STRATEGYQA_TRAIN)

    if dataset_size:
        random.seed(42)
        random.shuffle(data)
        data = data[:dataset_size]

    print(zero_shot_cot(dataset_name, data))

if __name__ == '__main__':
    Fire(main) 