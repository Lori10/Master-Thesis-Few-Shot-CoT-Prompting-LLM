import argparse
import json 
import random
import os
from utils import *

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
        "--random_seed", type=int, default=42, help="seed for selecting random samples"
    )

    parser.add_argument(
        "--labeled_demos_save_dir", type=str, default=42, help="directory to save the labeled demonstrations"
    )

    
    parser.add_argument(
        "--demos_save_dir", type=str, default="demos/", help="maximum number of reasoning chains"
    )

    parser.add_argument(
        "--dataset_size_limit", type=int, default=100, help="whether to limit training dataset size. if 0, the dataset size is unlimited and we use all the samples in the dataset for creating the demonstrations."
    )


    args = parser.parse_args()
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
    dataloader = create_dataloader(args, answers_available=True)

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