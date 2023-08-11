import os
from utils.load_data import create_dataloader 
import datetime 
import argparse
import json
import time

def parse_arguments():
    parser = argparse.ArgumentParser(description="Auto-CoT")
    parser.add_argument(
        "--dataset", type=str, default="aqua",
        choices=["aqua", "gsm8k", "commonsensqa", "addsub", "multiarith", "strategyqa", "svamp", "singleeq", "coin_flip", "last_letters"], help="dataset used for experiment"
    )

    parser.add_argument(
        "--data_path", type=str, default="../datasets/AQuA/train.json",
        choices=["../datasets/gsm8k/train.jsonl", "../datasets/AQuA/train.json"], help="dataset used for experiment"
    )

    parser.add_argument(
        "--dataset_size_limit", type=int, default=1000, help="whether to limit training dataset size. if 0, the dataset size is unlimited and we use all the samples in the dataset for creating the demonstrations."
    )

    parser.add_argument("--random_seed", type=int, default=42, help="random seed")

    parser.add_argument(
        "--answers_are_available", type=bool, default=True, help='true if answers are available in the test dataset, false otherwise'
    )

    parser.add_argument(
        "--output_dir", type=str, default='subset_data', help='directory to save the subset data in'
    )

    args = parser.parse_args()
    return args

def main():
    args = parse_arguments()
    
    current_time = datetime.datetime.now()
    time_string = current_time.strftime("%Y_%m_%d_%H_%M_%S")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        os.makedirs(args.output_dir + '/' + time_string)
    else:
        os.makedirs(args.output_dir + '/' + time_string)

    args.args_file = args.output_dir + '/' + time_string + '/args.json'
    args.output_dir = args.output_dir + '/' + time_string + '/'

    start = time.time()
    dataloader = create_dataloader(args)

    end = time.time()
    args_dict = {
        'dataset': args.dataset,
        'data_path': args.data_path,
        'dataset_size_limit': args.dataset_size_limit,
        'random_seed': args.random_seed,
        'answers_are_available': args.answers_are_available,
        'output_dir': args.output_dir,
        "execution_time": str(end - start) + ' seconds',
    }
    with open(args.args_file, 'w') as f:
        json.dump(args_dict, f, indent=4)

    demos = {"subset_data": dataloader}
    with open(args.output_dir + 'subset_data', 'w', encoding="utf-8") as write_f:
        json.dump(demos, write_f, indent=4, ensure_ascii=False)    

    print('Exporting subset data DONE!')

if __name__ == "__main__":
    main()