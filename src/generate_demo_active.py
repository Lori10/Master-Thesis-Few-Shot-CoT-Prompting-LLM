# This file used to generate uncertainty score for each question
import time
import argparse
import json
import load_env_vars
import datetime 
import os
from utils.load_data import create_dataloader
from utils.prompts_llm import build_prompt_initialize_llmchain
from utils.uncertainty_estimation import generate_uncertainty_all_questions

def arg_parser():
    parser = argparse.ArgumentParser(description="Active_CoT")
    parser.add_argument(
        "--dataset", type=str, default="aqua", choices=["gsm8k", "aqua"], help="dataset to inference"
    )

    parser.add_argument(
        "--data_path", type=str, default="../datasets/AQuA/train.json",
        choices=["../datasets/gsm8k/train.jsonl", "../datasets/AQuA/train.json"], help="dataset used for experiment"
    )
    
    parser.add_argument(
        "--dir_prompts", type=str, default="uncertainty_estimation_prompts/aqua", help="prompts to use"
    )

    parser.add_argument(
        "--model_id", type=str, default="text-davinci-003", choices=["text-davinci-003", "tiiuae/falcon-7b-instruct"], help="model used for decoding."
    )
    
    parser.add_argument(
        "--method", type=str, default="few_shot_cot", choices=["zero_shot_cot", "few_shot_cot"], help="method"
    )
    
    parser.add_argument(
        "--dataset_size_limit", type=int, default=5, help="whether to limit dataset size. if 0, the dataset size is unlimited and we use all the samples in the dataset for creating the demonstrations."
    )

    parser.add_argument("--random_seed", type=int, default=42, help="random seed")
    
    parser.add_argument(
        "--temperature", type=float, default=0.7, help="temperature for llm decoding"
    )
    parser.add_argument(
        "--num_trails", type=int, default=3, help="number of trails to run for each qeestion"
    )
    parser.add_argument(
        "--sort_by", type=str, default='entropy', choices=['disagreement', 'variance', 'entropy'], help="sort the final result by given option"
    )

    parser.add_argument(
        "--nr_demos", type=int, default=3, help='nr of demonstrations to select'
    )

    # use the sorted uncertainty file to select the demonstrations for Active CoT
    # aqua: uncertainties/aqua/2023_08_11_17_21_05/sorted_all_uncertainty_records'
    parser.add_argument(
        "--load_uncertainty_file", type=str, default=None, help='nr of demonstrations to select'
    )

    parser.add_argument(
        "--answers_are_available", type=bool, default=True, help='true if answers are available in the test dataset, false otherwise'
    )
    
    args = parser.parse_args()
    
    # Fill in the dataset path
    if args.dataset == "gsm8k":
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif args.dataset == "aqua":
        args.direct_answer_trigger = "\nThe answer is"
    else:
        raise ValueError("dataset is not properly defined ...")

    if args.answers_are_available:
        args.demos_save_dir = "labeled_demos"
    else:
        args.demos_save_dir = "unlabeled_demos"
        
    # "Therefore, the answer ..." -> "The answer ..."
    trigger = args.direct_answer_trigger.replace("\nTherefore, ", "")
    args.direct_answer_trigger_for_zeroshot = trigger[0].upper() + trigger[1:]
    args.direct_answer_trigger_for_zeroshot_cot = args.direct_answer_trigger
    args.direct_answer_trigger_for_fewshot = "The answer is"
    args.cot_trigger = "Let's think step by step."
    
    return args


def main():
    args = arg_parser()

    current_time = datetime.datetime.now()
    time_string = current_time.strftime("%Y_%m_%d_%H_%M_%S")
    if not os.path.exists(args.demos_save_dir):
        os.makedirs(args.demos_save_dir)
        os.makedirs(args.demos_save_dir + '/' + 'active')
        os.makedirs(args.demos_save_dir + '/' +  'active' + '/' + time_string)
        os.makedirs(args.demos_save_dir + '/' + 'active' + '/' + time_string + '/' + 'demos')
        os.makedirs(args.demos_save_dir + '/' + 'active' + '/' + time_string + '/' + 'uncertainty_scores')
    elif not os.path.exists(args.demos_save_dir + '/' + 'active'):
        os.makedirs(args.demos_save_dir + '/' + 'active')
        os.makedirs(args.demos_save_dir + '/' +  'active' + '/' + time_string)
        os.makedirs(args.demos_save_dir + '/' + 'active' + '/' + time_string + '/' + 'demos')
        os.makedirs(args.demos_save_dir + '/' + 'active' + '/' + time_string + '/' + 'uncertainty_scores')
    else:
        os.makedirs(args.demos_save_dir + '/' +  'active' + '/' + time_string)
        os.makedirs(args.demos_save_dir + '/' + 'active' + '/' + time_string + '/' + 'demos')
        os.makedirs(args.demos_save_dir + '/' + 'active' + '/' + time_string + '/' + 'uncertainty_scores')

    args.args_file = args.demos_save_dir + '/' + 'active' + '/' + time_string + '/' + 'args.json'
    args.uncertainty_scores_dir = args.demos_save_dir + '/' + 'active' + '/' + time_string + '/' + 'uncertainty_scores/'
    args.demos_save_dir = args.demos_save_dir + '/' + 'active' + '/' + time_string + '/' + 'demos/'
    
    dataloader = create_dataloader(args)

    start = time.time()

    if args.load_uncertainty_file: 
        with open(args.load_uncertainty_file, 'r', encoding="utf-8") as f:
            result = json.load(f)['result']
    else: 
        build_prompt_initialize_llmchain(args)
        result = generate_uncertainty_all_questions(args, dataloader, True)

    end = time.time()

    args_dict = {
        "sampling_method": "Active",
        "dataset": args.dataset,
        "data_path": args.data_path,
        "dataset_size_limit": args.dataset_size_limit,
        "random_seed": args.random_seed,
        "nr_demos": args.nr_demos,
        "demos_save_dir": args.demos_save_dir,
        "method": args.method,
        "model_id": args.model_id,
        "num_trails": args.num_trails,
        "sort_by": args.sort_by,
        "temperature": args.temperature,
        "answers_are_available": args.answers_are_available,
        "uncertainty_scores_dir": args.uncertainty_scores_dir,
        "dir_prompts": args.dir_prompts,
        "load_uncertainty_file": args.load_uncertainty_file,
        "execution_time": str(end - start) + " seconds",
    }

    with open(args.args_file, 'w') as f:
        json.dump(args_dict, f, indent=4)

    demos = {'demo': result[:args.nr_demos]}
    with open(f"{args.demos_save_dir}demos", 'w', encoding="utf-8") as write_f:
        json.dump(demos, write_f, indent=4, ensure_ascii=False)
        
    with open(args.uncertainty_scores_dir + 'uncertainties.txt', 'w') as f:
        try:
            f.write(json.dumps(result, indent=4))
        except:
            for item in result:
                try:
                    if args.dataset in ("gsm8k"):
                        f.write(f"{item}, uncertainty: {len(item[-1])}, variance: {item[1]}\n")
                    else:
                        f.write(f"{item}, uncertainty: {len(item[-1])}\n")
                except:
                    pass

    
    print('Active CoT finished!')

if __name__ == "__main__":
    main()