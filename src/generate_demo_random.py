import argparse
import json 
import random
import os
from utils.load_data import create_dataloader
import datetime
import time
from utils.filter_simple_examples import filter_examples_with_labels

def parse_arguments():
    parser = argparse.ArgumentParser(description="Random-CoT")
    parser.add_argument(
        "--dataset", type=str, default="gsm8k",
        choices=["aqua", "gsm8k"], help="dataset used for experiment"
    )

    parser.add_argument(
        "--data_path", type=str, default="../datasets/gpt35_zeroshotcot_training_data/gsm8k/QA_record_prompt1.txt",
        choices=["../datasets/original/gsm8k/train.jsonl", "../datasets/original/AQuA/train.json",
                 "../datasets/gpt35_zeroshotcot_training_data/gsm8k/QA_record_prompt1.txt",
                 "../datasets/gpt35_zeroshotcot_training_data/aqua/QA_record_prompt1.txt"], help="dataset used for experiment"
    )

    parser.add_argument(
        "--random_seed", type=int, default=1, help="seed for selecting random samples"
    )

    parser.add_argument(
        "--method_random_seed", type=int, default=1, help="seed for selecting random samples"
    )
    
    parser.add_argument(
        "--nr_seeds", type=int, default=2, help="nr of different prompts to select"
    )

    parser.add_argument(
        "--dataset_size_limit", type=int, default=1000, help="whether to limit training dataset size. if 0, the dataset size is unlimited and we use all the samples in the dataset for creating the demonstrations."
    )
    parser.add_argument(
        "--nr_demos", type=int, default=8, help="nr of demonstrations to select"
    )

    parser.add_argument(
        "--max_ra_len", type=int, default=12, help="maximum number of reasoning chains"
    )

    parser.add_argument(
        "--max_token_len", type=int, default=86, help="maximum number of reasoning chains"
    )

    parser.add_argument(
        "--answers_are_available", type=bool, default=True, help='true if answers are available in the test dataset, false otherwise'
    )

    args = parser.parse_args()
    if args.answers_are_available:
        args.demos_save_dir = "labeled_demos"
    else:
        args.demos_save_dir = "unlabeled_demos"
    return args

def main():
    args = parse_arguments()

    current_time = datetime.datetime.now()
    time_string = current_time.strftime("%Y_%m_%d_%H_%M_%S")
    if not os.path.exists(args.demos_save_dir):
        os.makedirs(args.demos_save_dir)
        os.makedirs(args.demos_save_dir + '/' + 'random')
        os.makedirs(args.demos_save_dir + '/' +  'random' + '/' + time_string)
        os.makedirs(args.demos_save_dir + '/' + 'random' + '/' + time_string + '/' + 'demos')
    elif not os.path.exists(args.demos_save_dir + '/' + 'random'):
        os.makedirs(args.demos_save_dir + '/' + 'random')
        os.makedirs(args.demos_save_dir + '/' +  'random' + '/' + time_string)
        os.makedirs(args.demos_save_dir + '/' + 'random' + '/' + time_string + '/' + 'demos')
    else:
        os.makedirs(args.demos_save_dir + '/' +  'random' + '/' + time_string)
        os.makedirs(args.demos_save_dir + '/' + 'random' + '/' + time_string + '/' + 'demos')

    args.args_file = args.demos_save_dir + '/' + 'random' + '/' + time_string + '/' + 'args.json'
    args.demos_save_dir = args.demos_save_dir + '/' + 'random' + '/' + time_string + '/' + 'demos/'

    args_dict = {
        "sampling_method": "Random",
        "dataset": args.dataset,
        "data_path": args.data_path,
        "dataset_size_limit": args.dataset_size_limit,
        "training_set_random_seed": args.random_seed,
        "method_random_seed": args.method_random_seed,
        "nr_seeds": args.nr_seeds,
        "nr_demos": args.nr_demos,
        "answers_are_available": args.answers_are_available,
        "demos_save_dir": args.demos_save_dir,
        "method_random_seed": args.method_random_seed
    }

    dataloader = create_dataloader(args)   

    if 'zeroshotcot' in args.data_path:
        dataloader = filter_examples_with_labels(args, dataloader, args.max_token_len, args.max_ra_len)
        args_dict["nr_filtered_examples"] = len(dataloader)
        args_dict['max_ra_len'] = args.max_ra_len
        args_dict['max_token_len'] = args.max_token_len

    random.seed(args.method_random_seed)

    start = time.time()  
    for i in range(args.nr_seeds):
        print(len(dataloader))
        selected_examples = random.sample(dataloader, args.nr_demos)
        demos = [example for example in selected_examples]
        demos_dic = {"demo": demos}
        with open(args.demos_save_dir + '/demos' + str(i+1), 'w', encoding="utf-8") as write_f:
            json.dump(demos_dic, write_f, indent=4, ensure_ascii=False)

    end = time.time()
    
    args_dict["execution_time"] = str(end - start) + ' seconds'

    with open(args.args_file, 'w') as f:
        json.dump(args_dict, f, indent=4)

    print('Random Demo Generation Finished!')

if __name__ == "__main__":
    main()