import time
import argparse
import json
import datetime 
import os
from utils.load_data import create_dataloader
from utils.uncertainty_estimation import generate_uncertainty_all_questions
from utils.prompts_llm import create_prompts_inference, initialize_llmchain, initialize_llm
import sys
import random

def arg_parser():
    parser = argparse.ArgumentParser(description="Active_CoT")
    parser.add_argument(
        "--dataset", type=str, default="gsm8k", choices=["gsm8k", "aqua"], help="dataset to inference"
    )

    parser.add_argument(
        "--data_path", type=str, default="../datasets/gsm8k/train.jsonl",
        choices=["../datasets/gsm8k/train.jsonl", "../datasets/AQuA/train.json"], help="dataset used for experiment"
    )
    
    parser.add_argument(
        "--dir_prompts", type=str, default="uncertainty_estimation_prompts/gsm8k", help="prompts to use"
    )

    parser.add_argument(
        "--model_id", type=str, default="gpt-35-turbo-0613", choices=["gpt-35-turbo-0613" ,"text-davinci-003", "tiiuae/falcon-7b-instruct"], help="model used for decoding."
    )
    
    parser.add_argument(
        "--method", type=str, default="cot", choices=["standard", "zero_shot_cot", "cot"], help="method"
    )

    parser.add_argument(
        "--temperature", type=float, default=0.7, help="temperature for llm decoding"
    )
    parser.add_argument(
        "--num_trails", type=int, default=5, help="number of trails to run for each qeestion"
    )
    parser.add_argument(
        "--sort_by", type=str, default='entropy', choices=['disagreement', 'variance', 'entropy'], help="sort the final result by given option"
    )
    
    parser.add_argument(
        "--dataset_size_limit", type=int, default=1000, help="whether to limit dataset size. if 0, the dataset size is unlimited and we use all the samples in the dataset for creating the demonstrations."
    )

    parser.add_argument("--random_seed", type=int, default=1, help="random seed")
    
    parser.add_argument(
        "--uncertainty_metric", type=str, default='disagreement', choices=['disagreement', 'variance', 'entropy'], help="sort the final result by given option"
    )

    parser.add_argument(
        "--uncertainty_metric_value", type=str, default=5, choices=['disagreement', 'variance', 'entropy'], help="sort the final result by given option"
    )

    parser.add_argument(
        "--nr_demos", type=int, default=8, help='nr of demonstrations to select'
    )

    # use the sorted uncertainty file to select the demonstrations for Active CoT
    parser.add_argument(
        "--load_uncertainty_file", type=str, default='final_uncertainties/2023_08_29_14_44_47/unsorted_all_uncertainty_records', help='nr of demonstrations to select'
    )

    parser.add_argument(
        "--load_uncertainty_args_file", type=str, default='final_uncertainties/2023_08_29_14_44_47/args.json', help='nr of demonstrations to select'
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

    args_dict = {
        'dataset': args.dataset,
        'data_path': args.data_path,
        'nr_demos': args.nr_demos,
        'uncertainty_metric': args.uncertainty_metric,
        'uncertainty_metric_value': args.uncertainty_metric_value,
        'load_uncertainty_file': args.load_uncertainty_file,
        'load_uncertainty_args_file': args.load_uncertainty_args_file,
        'answers_are_available': args.answers_are_available,
        'demos_save_dir': args.demos_save_dir,
        'uncertainty_scores_dir': args.uncertainty_scores_dir
    }

    if args.load_uncertainty_file and args.load_uncertainty_args_file: 
        with open(args.load_uncertainty_file, 'r', encoding="utf-8") as f:
            result = json.load(f)['result']

        with open(args.load_uncertainty_args_file, 'r', encoding="utf-8") as f:
            uncertainty_args = json.load(f)

        args_dict['generate_uncertainty_args'] = uncertainty_args

    else: 
        args_dict["method"] = args.method
        args_dict["model_id"] = args.model_id
        args_dict["num_trails"] = args.num_trails
        args_dict["sort_by"] = args.sort_by
        args_dict["temperature"] = args.temperature
        args_dict["dir_prompts"] = args.dir_prompts
        
        dataloader = create_dataloader(args)
        prompts_list = create_prompts_inference(args)

        assert len(prompts_list) == 1
        azure_llm = initialize_llm(args, is_azureopenai=True)
        azure_llm_chain = initialize_llmchain(azure_llm, prompts_list[0])
        openai_llm = initialize_llm(args, is_azureopenai=False)
        openai_llm_chain = initialize_llmchain(openai_llm, prompts_list[0])

        # print('PROMPT FOR UNCERTAINTY ESTIMATION:')
        # print(from_chatmodelmessages_to_string(args.llm_chain.prompt.messages))
        # print('*' * 50)

        result = generate_uncertainty_all_questions(args, dataloader, True, azure_llm_chain, openai_llm_chain)

    random.seed(args.random_seed)
    random.shuffle(result)

    selected_examples = []
    for example in result:
        if example[args.uncertainty_metric] == args.uncertainty_metric_value:
            selected_examples.append(example)

        if len(selected_examples) == args.nr_demos:
            break

    demos = {'demo': selected_examples}

    with open(f"{args.demos_save_dir}demos", 'w', encoding="utf-8") as write_f:
        json.dump(demos, write_f, indent=4, ensure_ascii=False)

    with open(args.args_file, 'w') as f:
        json.dump(args_dict, f, indent=4)

    with open(args.uncertainty_scores_dir + 'uncertainties.txt', 'w') as f:
        try:
            f.write(json.dumps(selected_examples, indent=4))
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