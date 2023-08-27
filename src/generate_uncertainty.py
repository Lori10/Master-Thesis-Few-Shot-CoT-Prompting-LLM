import argparse
import json 
import os
import datetime
import time
from utils.prompts_llm import create_prompts_inference, initialize_llmchain
from utils.uncertainty_estimation import generate_uncertainty_all_questions, sort_uncertainty
from utils.load_data import create_dataloader
import load_env_vars

def parse_arguments():
    parser = argparse.ArgumentParser(description="Uncertainty-Estimator")
    parser.add_argument(
        "--dataset", type=str, default="gsm8k",
        choices=["aqua", "gsm8k", "commonsensqa", "addsub", "multiarith", "strategyqa", "svamp", "singleeq", "coin_flip", "last_letters"], help="dataset used for experiment"
    )

    parser.add_argument(
        "--data_path", type=str, default="../datasets/gsm8k/train.jsonl",
        choices=["../datasets/gsm8k/train.jsonl", "../datasets/AQuA/train.json"], help="dataset used for experiment"
    )

    parser.add_argument(
        "--random_seed", type=int, default=1, help="seed for selecting random samples"
    )

    parser.add_argument(
        "--dataset_size_limit", type=int, default=1000, help="whether to limit training dataset size. if 0, the dataset size is unlimited and we use all the samples in the dataset for creating the demonstrations."
    )

    parser.add_argument(
        "--sort_by", type=str, default='entropy', choices=['disagreement', 'variance', 'entropy'], help="sort the final result by given option"
    )

    parser.add_argument(
        "--num_trails", type=int, default=5, help="number of trails to run for each qeestion"
    )

    parser.add_argument(
        "--temperature", type=float, default=0.7, help="temperature for llm decoding"
    )

    parser.add_argument(
        "--model_id", type=str, default="gpt-35-turbo-0613", choices=["gpt-35-turbo-0613" ,"text-davinci-003", "tiiuae/falcon-7b-instruct"], help="model used for decoding."
    )

    parser.add_argument(
        "--method", type=str, default="cot", choices=["zero_shot_cot", "standard", "cot"], help="method"
    )

    parser.add_argument(
        "--dir_prompts", type=str, default="uncertainty_estimation_prompts/gsm8k", help="prompts to use"
    )

    parser.add_argument(
        "--answers_are_available", type=bool, default=True, help='true if answers are available in the test dataset, false otherwise'
    )

    parser.add_argument(
        "--uncertainty_save_dir", type=str, default="uncertainties", help="output directory"
    )

    args = parser.parse_args()

    if args.dataset == "gsm8k":
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif args.dataset == "aqua":
        args.direct_answer_trigger = "\nThe answer is"
    else:
        raise ValueError("dataset is not properly defined ...")

    trigger = args.direct_answer_trigger.replace("\nTherefore, ", "")
    args.direct_answer_trigger_for_zeroshot = trigger[0].upper() + trigger[1:]
    args.direct_answer_trigger_for_zeroshot_cot = args.direct_answer_trigger
    args.direct_answer_trigger_for_fewshot = "The answer is"
    args.cot_trigger = "Let's think step by step."

    return args

def main(): 
    args = parse_arguments()

    current_time = datetime.datetime.now()
    time_string = current_time.strftime("%Y_%m_%d_%H_%M_%S")
    if not os.path.exists(args.uncertainty_save_dir):
        os.makedirs(args.uncertainty_save_dir)
        os.makedirs(args.uncertainty_save_dir + '/' + args.dataset)
        os.makedirs(args.uncertainty_save_dir + '/' + args.dataset + '/' + time_string)
    elif not os.path.exists(args.uncertainty_save_dir + '/' + args.dataset):
        os.makedirs(args.uncertainty_save_dir + '/' + args.dataset)
        os.makedirs(args.uncertainty_save_dir + '/' + args.dataset + '/' + time_string)
    else:
        os.makedirs(args.uncertainty_save_dir + '/' + args.dataset + '/' + time_string)

    args.uncertainty_save_dir = args.uncertainty_save_dir + '/' + args.dataset + '/' + time_string + '/'
    args.args_file = args.uncertainty_save_dir + 'args.json'

    dataloader = create_dataloader(args)

    start = time.time()

    prompts_list = create_prompts_inference(args)
    assert len(prompts_list) == 1

    initialize_llmchain(args, prompts_list[0], llm_init=False)
    result = generate_uncertainty_all_questions(args, dataloader, False)
    end = time.time()
    
    args_dict = {
        'dataset': args.dataset,
        'data_path': args.data_path,
        'random_seed': args.random_seed,
        'dataset_size_limit': args.dataset_size_limit,
        'num_trails': args.num_trails,
        'temperature': args.temperature,
        'model_id': args.model_id,
        'method': args.method,
        'dir_prompts': args.dir_prompts,
        'answers_are_available': args.answers_are_available,
        'sort_by': args.sort_by,
        'uncertainty_save_dir': args.uncertainty_save_dir,
        'execution_time': str(end - start) + ' seconds',
    }

    with open(args.args_file, 'w') as f:
        json.dump(args_dict, f, indent=4)

    unsorted_result = {'result': result}
    with open(f"{args.uncertainty_save_dir}unsorted_all_uncertainty_records", 'w', encoding="utf-8") as write_f:
        json.dump(unsorted_result, write_f, indent=2, ensure_ascii=False)

    sorted_result = sort_uncertainty(args, result)
    sorted_result = {'result': sorted_result}
    with open(f"{args.uncertainty_save_dir}sorted_all_uncertainty_records", 'w', encoding="utf-8") as write_f:
        json.dump(sorted_result, write_f, indent=2, ensure_ascii=False)
    
    print('Uncertainty estimation finished!')

if __name__ == '__main__':
    main()