import argparse
import json 
import random
import os
from utils import *

def parse_arguments():
    parser = argparse.ArgumentParser(description="Random-CoT")
    parser.add_argument(
        "--dataset", type=str, default="aqua",
        choices=["aqua", "gsm8k", "commonsensqa", "addsub", "multiarith", "strategyqa", "svamp", "singleeq", "coin_flip", "last_letters"], help="dataset used for experiment"
    )

    parser.add_argument(
        "--data_path", type=str, default="../datasets/AQuA/train.json",
        choices=["../datasets/gsm8k/train.jsonl", "../datasets/AQuA/train.json"], help="dataset used for experiment"
    )

    parser.add_argument(
        "--random_seed", type=int, default=42, help="seed for selecting random samples"
    )

    parser.add_argument(
        "--nr_seeds", type=int, default=2, help="nr of different prompts to select"
    )

    # parser.add_argument(
    #     "--labeled_demos_save_dir", type=str, default=42, help="directory to save the labeled demonstrations"
    # )

    parser.add_argument(
        "--dataset_size_limit", type=int, default=100, help="whether to limit training dataset size. if 0, the dataset size is unlimited and we use all the samples in the dataset for creating the demonstrations."
    )
    parser.add_argument(
        "--nr_demos", type=int, default=5, help="nr of demonstrations to select"
    )

    parser.add_argument(
        "--answers_are_available", type=bool, default=True, help='true if answers are available in the test dataset, false otherwise'
    )


    args = parser.parse_args()
    if args.answers_are_available:
        args.demos_save_dir = "labeled_demos/"
    else:
        args.demos_save_dir = "unlabeled_demos/"
    return args

def main():
    args = parse_arguments()

    if not os.path.exists(args.demos_save_dir):
        os.makedirs(args.demos_save_dir)
        os.makedirs(args.demos_save_dir + 'random')
        os.makedirs(args.demos_save_dir + 'random/' + args.dataset)
    elif not os.path.exists(args.demos_save_dir + 'random'):
        os.makedirs(args.demos_save_dir + 'random')
        os.makedirs(args.demos_save_dir + 'random/' + args.dataset)
    elif not os.path.exists(args.demos_save_dir + 'random/' + args.dataset):
        os.makedirs(args.demos_save_dir + 'random/' + args.dataset)

    args.demos_save_dir = f"{args.demos_save_dir}/random/{args.dataset}/"

    random.seed(args.random_seed)
    dataloader = create_dataloader(args)

    if args.dataset_size_limit <= 0:
        args.dataset_size_limit = len(dataloader)
    else:
        dataloader = dataloader[:args.dataset_size_limit] # replace 7 with 1000; only take 1000 questions randomly to annotate, randomness decided by seed
    print(f"Dataloader size: {len(dataloader)}")

    for i in range(args.nr_seeds):
        selected_examples = random.sample(dataloader, args.nr_demos)
        demos = [example for example in selected_examples]
        demos_dic = {"demo": demos}
        with open(args.demos_save_dir + 'demos' + str(i+1), 'w', encoding="utf-8") as write_f:
            json.dump(demos_dic, write_f, indent=4, ensure_ascii=False)
            
if __name__ == "__main__":
    main()