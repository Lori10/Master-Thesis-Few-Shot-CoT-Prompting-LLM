import argparse
import json 
import random
import os
from utils import *
import datetime

def parse_arguments():
    parser = argparse.ArgumentParser(description="Random-CoT")
    parser.add_argument(
        "--dataset", type=str, default="gsm8k",
        choices=["aqua", "gsm8k", "commonsensqa", "addsub", "multiarith", "strategyqa", "svamp", "singleeq", "coin_flip", "last_letters"], help="dataset used for experiment"
    )

    parser.add_argument(
        "--data_path", type=str, default="../datasets/gsm8k/train.jsonl",
        choices=["../datasets/gsm8k/train.jsonl", "../datasets/AQuA/train.json"], help="dataset used for experiment"
    )

    parser.add_argument(
        "--random_seed", type=int, default=22, help="seed for selecting random samples"
    )

    parser.add_argument(
        "--nr_seeds", type=int, default=2, help="nr of different prompts to select"
    )

    parser.add_argument(
        "--dataset_size_limit", type=int, default=70, help="whether to limit training dataset size. if 0, the dataset size is unlimited and we use all the samples in the dataset for creating the demonstrations."
    )
    parser.add_argument(
        "--nr_demos", type=int, default=6, help="nr of demonstrations to select"
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

    args.json_file = args.demos_save_dir + '/' + 'random' + '/' + time_string + '/' + 'args.json'
    args.demos_save_dir = args.demos_save_dir + '/' + 'random' + '/' + time_string + '/' + 'demos/'
    args_dict = {
        "sampling_method": "Random",
        "dataset": args.dataset,
        "data_path": args.data_path,
        "dataset_size_limit": args.dataset_size_limit,
        "random_seed": args.random_seed,
        "nr_seeds": args.nr_seeds,
        "nr_demos": args.nr_demos,
        "answers_are_available": args.answers_are_available,
        "demos_save_dir": args.demos_save_dir
    }

    with open(args.json_file, 'w') as f:
        json.dump(args_dict, f, indent=4)

    print('Hyperparameters:')

    random.seed(args.random_seed)
    dataloader = create_dataloader(args)

    if args.dataset_size_limit <= 0:
        args.dataset_size_limit = len(dataloader)
    else:
        dataloader = dataloader[:args.dataset_size_limit] # replace 7 with 1000; only take 1000 questions randomly to annotate, randomness decided by seed
    print(f"Proceeding with data size: {len(dataloader)}")

    print(f'nr of seeds: {args.nr_seeds}')
    print(f'nr of demos: {args.nr_demos}')
    print(f'random seed: {args.random_seed}')
    
    for i in range(args.nr_seeds):
        selected_examples = random.sample(dataloader, args.nr_demos)
        demos = [example for example in selected_examples]
        demos_dic = {"demo": demos}
        with open(args.demos_save_dir + '/demos' + str(i+1), 'w', encoding="utf-8") as write_f:
            json.dump(demos_dic, write_f, indent=4, ensure_ascii=False)
            
if __name__ == "__main__":
    main()