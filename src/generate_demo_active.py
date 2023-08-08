# This file used to generate uncertainty score for each question
from utils import *
import time
import argparse
import json
from utils import initialize_llmchain
import load_env_vars
from constant_vars import *
import datetime 
from utils import *

def arg_parser():
    parser = argparse.ArgumentParser(description="Active_CoT")
    parser.add_argument("--random_seed", type=int, default=42, help="random seed")
    parser.add_argument(
        "--dataset", type=str, default="aqua", choices=["gsm8k", "aqua"], help="dataset to inference"
    )

    parser.add_argument(
        "--data_path", type=str, default="../datasets/AQuA/train.json",
        choices=["../datasets/gsm8k/train.jsonl", "../datasets/AQuA/train.json"], help="dataset used for experiment"
    )
    
    parser.add_argument(
        "--dir_prompts", type=str, default="uncertainty_estimation_prompts/gsm8k", help="prompts to use"
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

    parser.add_argument(
        "--answers_are_available", type=bool, default=True, help='true if answers are available in the test dataset, false otherwise'
    )

    parser.add_argument(
        "--uncertainty_scores_dir", type=str, default='uncertainty_scores/', help='directory where the uncertainty scores are saved'
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

    args.json_file = args.demos_save_dir + '/' + 'active' + '/' + time_string + '/' + 'args.json'
    args.uncertainty_scores_dir = args.demos_save_dir + '/' + 'active' + '/' + time_string + '/' + 'uncertainty_scores/'
    args.demos_save_dir = args.demos_save_dir + '/' + 'active' + '/' + time_string + '/' + 'demos/'

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
    }

    with open(args.json_file, 'w') as f:
        json.dump(args_dict, f, indent=4)


    if args.dataset == "gsm8k":
        prefix = prefix_gsm8k
    elif args.dataset == "aqua":
        prefix = prefix_aqua
    else:
        raise NotImplementedError("dataset not implemented")
    
    print('Hyperparameters: ')

    set_random_seed(args.random_seed)
    dataloader = create_dataloader(args)

    if args.dataset_size_limit <= 0:
        args.dataset_size_limit = len(dataloader)
    else:
        dataloader = dataloader[:args.dataset_size_limit] # replace 7 with 1000; only take 1000 questions randomly to annotate, randomness decided by seed
    print(f"Proceeding with data size: {len(dataloader)}")
    print(f"model_id: {args.model_id}")
    print(f"method: {args.method}")
    print(f"num_trails: {args.num_trails}")
    print(f"sort_by: {args.sort_by}")
    print(f"temperature: {args.temperature}")
    print(f"random_seed: {args.random_seed}")
    print(f"nr_demos: {args.nr_demos}")
    
    if args.method == "few_shot_cot":
        args.prefix = prefix + ' Follow the format of the examples below:\n'
        given_prompt_list = create_several_input_prompts(args, cot_flag=True)
        assert len(given_prompt_list) == 1
        args.prompt = given_prompt_list[0]
    elif args.method == "zero_shot_cot":
        args.prompt = prefix + "\nQ: " + "{question}" + "\nA: Let's think step by step."
    
    print(f'PROMPT:\n{args.prompt}\n')
    args.llm_chain = initialize_llmchain(args.prompt, args)

    start =time.time()  
    result = create_uncertainty(args, dataloader)
    end = time.time()
    print('Total Execution Time: ', end - start, " seconds")

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


if __name__ == "__main__":
    main()