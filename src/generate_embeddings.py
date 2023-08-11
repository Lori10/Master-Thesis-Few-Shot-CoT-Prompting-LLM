import numpy as np
import json
import argparse
import os
import load_env_vars
import datetime 
import pickle
import time
from utils.load_data import create_dataloader
from utils.embedding_generation import generate_corpus_embeddings

def parse_arguments():
    parser = argparse.ArgumentParser(description="Embeddings-Generator")
    parser.add_argument(
        "--dataset", type=str, default="aqua",
        choices=["aqua", "gsm8k", "commonsensqa", "addsub", "multiarith", "strategyqa", "svamp", "singleeq", "coin_flip", "last_letters"], help="dataset used for experiment"
    )

    parser.add_argument(
        "--data_path", type=str, default="../datasets/AQuA/train.json",
        choices=["../datasets/gsm8k/train.jsonl", "../datasets/AQuA/train.json"], help="dataset used for experiment"
    )

    parser.add_argument("--random_seed", type=int, default=42, help="random seed")

    parser.add_argument(
        "--dataset_size_limit", type=int, default=20, help="whether to limit training dataset size. if 0, the dataset size is unlimited and we use all the samples in the dataset for creating the demonstrations."
    )

    parser.add_argument(
        "--output_dir", type=str, default='embeddings', help="the directory where embeddings will be stored"
    )

    args = parser.parse_args()
    return args

def main():
    args = parse_arguments()

    current_time = datetime.datetime.now()
    time_string = current_time.strftime("%Y_%m_%d_%H_%M_%S")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        os.makedirs(args.output_dir + '/' + args.dataset)
        os.makedirs(args.output_dir + '/' + args.dataset + '/' + time_string)
    elif not os.path.exists(args.output_dir + '/' + args.dataset):
        os.makedirs(args.output_dir + '/' + args.dataset)
        os.makedirs(args.output_dir + '/' + args.dataset + '/' + time_string)
    else:
        os.makedirs(args.output_dir + '/' + args.dataset + '/' + time_string)
        
    args.output_dir = args.output_dir + '/' + args.dataset + '/' + time_string + '/'
    args.args_file = args.output_dir + 'args.json'

    start = time.time()
    
    args.answers_are_available = False
    dataloader = create_dataloader(args)
    corpus_embeddings = generate_corpus_embeddings(args, dataloader)

    end = time.time()
    args_dict = {
        'dataset': args.dataset,
        'data_path': args.data_path,
        'random_seed': args.random_seed,
        'dataset_size_limit': args.dataset_size_limit,
        'output_dir': args.output_dir,
        'execution_time': str(end - start) + ' seconds',
    }

    with open(args.args_file, 'w') as f:
        json.dump(args_dict, f, indent=4)

    with open(f"{args.output_dir}embeddings.pkl", "wb") as f:
        pickle.dump(corpus_embeddings, f)

    np.save(f"{args.output_dir}embeddings.npy", corpus_embeddings)
    print('Embeddings saved!')

if __name__ == "__main__":
    main()