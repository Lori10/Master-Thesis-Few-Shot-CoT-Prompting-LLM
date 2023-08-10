import numpy as np
import json
import argparse
from langchain.embeddings import OpenAIEmbeddings
import os
from utils import *
import load_env_vars
import openai
import datetime 
import pickle

def parse_arguments():
    parser = argparse.ArgumentParser(description="Embeddings-Generator")
    parser.add_argument(
        "--dataset", type=str, default="gsm8k",
        choices=["aqua", "gsm8k", "commonsensqa", "addsub", "multiarith", "strategyqa", "svamp", "singleeq", "coin_flip", "last_letters"], help="dataset used for experiment"
    )

    parser.add_argument(
        "--data_path", type=str, default="../datasets/gsm8k/train.jsonl",
        choices=["../datasets/gsm8k/train.jsonl", "../datasets/AQuA/train.json"], help="dataset used for experiment"
    )

    parser.add_argument("--random_seed", type=int, default=42, help="random seed")

    parser.add_argument(
        "--dataset_size_limit", type=int, default=20, help="whether to limit training dataset size. if 0, the dataset size is unlimited and we use all the samples in the dataset for creating the demonstrations."
    )

    parser.add_argument(
        "--output_dir", type=str, default='embeddings', help="the directory where embeddings will be stored"
    )
    
    parser.add_argument(
        "--answers_are_available", type=bool, default=False, help='true if answers are available in the test dataset, false otherwise'
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

    args_dict = {
        'dataset': args.dataset,
        'data_path': args.data_path,
        'random_seed': args.random_seed,
        'dataset_size_limit': args.dataset_size_limit,
        'output_dir': args.output_dir
    }

    with open(args.args_file, 'w') as f:
        json.dump(args_dict, f, indent=4)

    set_random_seed(args.random_seed)
    dataloader = create_dataloader(args)

    if args.dataset_size_limit <= 0:
        args.dataset_size_limit = len(dataloader)
    else:
        dataloader = dataloader[:args.dataset_size_limit] 

    corpus = [example['question'] for example in dataloader]
    openai.api_key = os.getenv("OPENAI_API_KEY")
    headers = {
        "x-api-key": openai.api_key,
    }
    encoder = OpenAIEmbeddings(
        deployment="text-embedding-ada-002-v2", headers=headers, chunk_size=1
    )
    
    start = time.time()
    embeddings = np.array(encoder.embed_documents(corpus))

    end = time.time()
    print('Total Execution Time: ', end - start, " seconds")

    with open(f"{args.output_dir}embeddings.pkl", "wb") as f:
        pickle.dump(embeddings, f)

    np.save(f"{args.output_dir}embeddings.npy", embeddings)
    print('Embeddings saved!')

if __name__ == "__main__":
    main()