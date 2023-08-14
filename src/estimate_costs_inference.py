import argparse
import json
import datetime
import os
from utils.estimate_costs import all_prompts_costs, uncertainty_cost_all_examples, embedding_cost
from utils.load_data import create_dataloader
from utils.prompts_llm import build_prefix, create_prompts_inference, build_prompt_template
import load_env_vars

def arg_parser():
    parser = argparse.ArgumentParser(description="CoT")
    parser.add_argument("--random_seed", type=int, default=1, help="random seed")
    parser.add_argument(
        "--dataset", type=str, default="aqua", choices=["gsm8k", "aqua"], help="dataset to inference"
    )

    parser.add_argument(
        "--data_path", type=str, default="../datasets/AQuA/test.json", choices=["../datasets/AQuA/test.json", "../datasets/gsm8k/test.jsonl", "../datasets/gsm8k/train.jsonl", "../datasets/AQuA/train.json"], help="dataset to inference"
    )

    parser.add_argument(
        "--dataset_size_limit", type=int, default=0, help="size of dataset to inference"
    )

    parser.add_argument(
        "--dir_prompts", type=str, default="labeled_demos/random/2023_08_14_14_24_43/demos", help="prompts to use"
    )
    parser.add_argument(
        "--model_id", type=str, default="text-davinci-003", choices=["gpt-35-turbo-0613", "text-davinci-003", "tiiuae/falcon-7b-instruct"], help="model used for decoding."
    )

    parser.add_argument(
        "--method", type=str, default="zero_shot_cot", choices=["zero_shot_cot", "standard", "cot", "zero_shot_cot", "few_shot_cot"], help="method"
    )

    parser.add_argument(
        "--output_dir", type=str, default="estimated_costs", help="output directory"
    )

    parser.add_argument(
        "--num_trails", type=int, default=5, help="number of trails to run for each qeestion"
    )

    parser.add_argument(
        "--answers_are_available", type=bool, default=True, help='true if answers are available in the test dataset, false otherwise'
    )

    parser.add_argument(
        "--embedding_model_id", type=str, default="text-embedding-ada-002", help="the id of the embedding model to use"
    )

    parser.add_argument(
        "--type", type=str, default='inference', choices=['estimation', 'embedding', 'inference'], help="self-consistency path num"
    )
   
    args = parser.parse_args()
    
    if args.dataset == "gsm8k":
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif args.dataset == "aqua":
        args.direct_answer_trigger = "The answer is"
    else:
        raise ValueError("dataset is not properly defined ...")
        
    trigger = args.direct_answer_trigger.replace("\nTherefore, ", "")
    args.direct_answer_trigger_for_zeroshot = trigger[0].upper() + trigger[1:]
    args.direct_answer_trigger_for_zeroshot_cot = args.direct_answer_trigger
    args.direct_answer_trigger_for_fewshot = "The answer is"
    args.cot_trigger = "Let's think step by step."
    
    return args


def main():
    # load arguments from terminal
    args = arg_parser()

    current_time = datetime.datetime.now()
    time_string = current_time.strftime("%Y_%m_%d_%H_%M_%S")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        os.makedirs(args.output_dir + '/' + time_string)
    else:
        os.makedirs(args.output_dir + '/' + time_string)

    args.args_file = args.output_dir + '/' + time_string + '/args.json'
        
    dataloader = create_dataloader(args)

    args_dict = {
        'dataset': args.dataset,
        'data_path': args.data_path,
        'dataset_size_limit': args.dataset_size_limit,
        'random_seed': args.random_seed
    }

    if args.type == 'estimation':
        build_prefix(args)
        build_prompt_template(args)
        uncertainty_cost = uncertainty_cost_all_examples(args, dataloader)
        args_dict['dir_prompts'] = args.dir_prompts
        args_dict['model_id'] = args.model_id,
        args_dict['method'] = args.method
        args_dict['num_trails'] = args.num_trails
        args_dict['uncertainty_cost'] = uncertainty_cost

    elif args.type == 'inference':
        build_prefix(args)
        prompts_list = create_prompts_inference(args)
        costs_prompts = all_prompts_costs(args, dataloader, prompts_list)
        print(f'Costs prompts: {costs_prompts}')
        inference_cost = sum(costs_prompts)
        args_dict['dir_prompts'] = args.dir_prompts
        args_dict['model_id'] = args.model_id,
        args_dict['method'] = args.method
        args_dict['inference_cost'] = inference_cost
        
    elif args.type == 'embedding':
        embedding_costs = embedding_cost(args, dataloader)
        args_dict['embedding_model_id'] = args.embedding_model_id
        args_dict['embedding_cost'] = embedding_costs

    with open(args.args_file, 'w') as f:
        json.dump(args_dict, f, indent=4)

    print('Cost Estimation finished!')
    
if __name__ == "__main__":
    main()