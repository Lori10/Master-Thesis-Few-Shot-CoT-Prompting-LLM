import time
import argparse
import json
from constant_vars import *
import datetime
import os 
from utils.load_data import create_dataloader
from utils.prompts_llm import create_prompts_inference, initialize_llm, from_chatmodelmessages_to_string
from utils.save_results import inference_save_info
from utils.inference_llm import all_prompts_inference_opensource
import sys 

def arg_parser():
    parser = argparse.ArgumentParser(description="CoT")
    parser.add_argument("--random_seed", type=int, default=1, help="random seed")
    parser.add_argument(
        "--dataset", type=str, default="aqua", choices=["gsm8k", "aqua"], help="dataset to inference"
    )

    parser.add_argument(
        "--data_path", type=str, default="../datasets/AQuA/test.json", choices=["../datasets/AQuA/test.json", "../datasets/gsm8k/test.jsonl"], help="dataset to inference"
    )

    parser.add_argument(
        "--dir_prompts", type=str, default="labeled_demos/active/2023_08_30_12_28_41/demos", help="prompts to use"
    )
    parser.add_argument(
        "--model_id", type=str, default="vilsonrodrigues/falcon-7b-instruct-sharded", choices=["gpt-35-turbo-0613", "text-davinci-003", "mosaicml/mpt-7b-instruct", "tiiuae/falcon-7b-instruct"], help="model used for decoding."
    )

    parser.add_argument(
        "--method", type=str, default="cot", choices=["zero_shot_cot", "standard", "cot"], help="method"
    )

    parser.add_argument(
        "--output_dir", type=str, default="inference_results", help="output directory"
    )
    
    parser.add_argument(
        "--dataset_size_limit", type=int, default=0, help="size of dataset to inference"
    )
  
    parser.add_argument(
        "--temperature", type=float, default=0.0, help="temperature used for llm decoding"
    )
    parser.add_argument(
        "--multipath", type=int, default=1, help="self-consistency path num"
    )

    parser.add_argument(
        "--answers_are_available", type=bool, default=True, help='true if answers are available in the test dataset, false otherwise'
    )
   
    args = parser.parse_args()

    if args.multipath > 1:
        args.temperature = 0.7
    
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

    args.output_dir = args.output_dir + '/' + time_string + '/'
    args.args_file = args.output_dir + 'args.json'
        
    dataloader = create_dataloader(args)

    prompts_list = create_prompts_inference(args)
    
    llm = initialize_llm(args, opensource_llm=True)
    print('LLM: ')
    print(llm)

    if args.multipath != 1:
        print("Self-consistency Enabled, output each inference result is not available")
    
    start = time.time()

    correct_list, wrong_list, QA_record_list = all_prompts_inference_opensource(args, dataloader, prompts_list, llm)
    assert len(correct_list) == len(wrong_list) == len(QA_record_list)

    end = time.time()

    args_dict = {
                "dataset": args.dataset,
                "dataset_size_limit": args.dataset_size_limit,
                "data_path": args.data_path,
                "random_seed": args.random_seed,
                "model_id": args.model_id,
                "method": args.method,
                "output_dir": args.output_dir,
                "temperature": args.temperature,
                "multipath": args.multipath,
                "answers_are_available": args.answers_are_available,
                "execution_time": str(end - start) + ' seconds',
                }

    if args.method in ['cot', 'standard']:
        args_dict["dir_prompts"] = args.dir_prompts

    with open(args.args_file, 'w') as f:
        json.dump(args_dict, f, indent=4)

    if args.model_id.startswith("gpt-35"):
        prompts_list = [from_chatmodelmessages_to_string(prompt_messages.messages) for prompt_messages in prompts_list]
    else:
        prompts_list = [prompt_template.template for prompt_template in prompts_list]

    inference_save_info(args, correct_list, wrong_list, QA_record_list, prompts_list, len(dataloader))
    print('Inference finished!')


if __name__ == "__main__":
    main()
