import random
# from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np
import json
import matplotlib.pyplot as plt
import argparse
import os
from utils.load_data import create_dataloader 
from utils.embedding_generation import generate_corpus_embeddings
import datetime 
import pickle
import time

def parse_arguments():
    parser = argparse.ArgumentParser(description="Auto-CoT")
    parser.add_argument(
        "--dataset", type=str, default="gsm8k",
        choices=["aqua", "gsm8k"], help="dataset used for experiment"
    )

    parser.add_argument(
        "--data_path", type=str, default="../datasets/gsm8k/train.jsonl",
        choices=["../datasets/gsm8k/train.jsonl", "../datasets/AQuA/train.json"], help="dataset used for experiment"
    )
    parser.add_argument("--random_seed", type=int, default=1, help="random seed")
    
    parser.add_argument(
        "--sampling", type=str, default="center", help="whether to sample the cluster center first"
    )

    parser.add_argument(
        "--dataset_size_limit", type=int, default=1000, help="whether to limit training dataset size. if 0, the dataset size is unlimited and we use all the samples in the dataset for creating the demonstrations."
    )

    parser.add_argument(
        "--nr_demos", type=int, default=8, help="nr of demonstrations to select"
    )

    parser.add_argument(
        "--max_token_len", type=float, default=float('inf'), help="maximum number of reasoning chains"
    )

    parser.add_argument(
        "--max_ra_len", type=float, default=float('inf'), help="maximum number of reasoning chains"
    )
    
    parser.add_argument(
        "--answers_are_available", type=bool, default=True, help='true if answers are available in the test dataset, false otherwise'
    )

    parser.add_argument(
        "--embedding_model_id", type=str, default="text-embedding-ada-002-v2", help="the id of the embedding model to use"
    )

    parser.add_argument(
        "--load_embeddings_file", type=str, default='embeddings/gsm8k/2023_08_29_22_56_01/embeddings.pkl', help='file to load embeddings from; either None or a path to a file'
    )

    parser.add_argument(
        "--load_embeddings_args_file", type=str, default='embeddings/gsm8k/2023_08_29_22_56_01/args.json', help='file to load embeddings from; either None or a path to a file'
    )

    args = parser.parse_args()
    if args.answers_are_available:
        args.demos_save_dir = "labeled_demos"
    else:
        args.max_ra_len = 'None'
        args.demos_save_dir = "unlabeled_demos"
    return args

def main():
    args = parse_arguments()
    
    current_time = datetime.datetime.now()
    time_string = current_time.strftime("%Y_%m_%d_%H_%M_%S")
    if not os.path.exists(args.demos_save_dir):
        os.makedirs(args.demos_save_dir)
        os.makedirs(args.demos_save_dir + '/' + 'auto')
        os.makedirs(args.demos_save_dir + '/' +  'auto' + '/' + time_string)
        os.makedirs(args.demos_save_dir + '/' + 'auto' + '/' + time_string + '/' + 'demos')
        for i in range(args.nr_demos):
            os.makedirs(args.demos_save_dir + '/' + 'auto' + '/' + time_string + '/' + 'demos' + '/cluster_' + str(i))
    elif not os.path.exists(args.demos_save_dir + '/' + 'auto'):
        os.makedirs(args.demos_save_dir + '/' + 'auto')
        os.makedirs(args.demos_save_dir + '/' +  'auto' + '/' + time_string)
        os.makedirs(args.demos_save_dir + '/' + 'auto' + '/' + time_string + '/' + 'demos')
        for i in range(args.nr_demos):
            os.makedirs(args.demos_save_dir + '/' + 'auto' + '/' + time_string + '/' + 'demos' + '/cluster_' + str(i))
    else:
        os.makedirs(args.demos_save_dir + '/' +  'auto' + '/' + time_string)
        os.makedirs(args.demos_save_dir + '/' + 'auto' + '/' + time_string + '/' + 'demos')
        for i in range(args.nr_demos):
            os.makedirs(args.demos_save_dir + '/' + 'auto' + '/' + time_string + '/' + 'demos' + '/cluster_' + str(i))

    args.args_file = args.demos_save_dir + '/' + 'auto' + '/' + time_string + '/' + 'args.json'
    args.demos_save_dir = args.demos_save_dir + '/' + 'auto' + '/' + time_string + '/'

    
    args_dict = {
        "sampling_method": "Auto",
        "dataset": args.dataset,
        "data_path": args.data_path,
        "max_ra_len": args.max_ra_len,
        "max_token_len": args.max_token_len,
        "random_seed": args.random_seed,
        "sampling": args.sampling,
        "dataset_size_limit": args.dataset_size_limit,
        "nr_demos": args.nr_demos,
        "answers_are_available": args.answers_are_available,
        "load_embeddings_file": args.load_embeddings_file,
        "load_embeddings_args_file": args.load_embeddings_args_file,
        "demos_save_dir": args.demos_save_dir,
    }

    dataloader = create_dataloader(args)

    start = time.time()
    if args.load_embeddings_file and args.load_embeddings_args_file:
        with open(args.load_embeddings_file, 'rb') as read_f:
            corpus_embeddings = pickle.load(read_f)

        with open(args.load_embeddings_args_file, 'r', encoding="utf-8") as f:
            embeddings_args = json.load(f)

        args_dict['generate_embeddings_args'] = embeddings_args
    else:
        args_dict['embedding_model_id'] = args.embedding_model_id
        corpus_embeddings = generate_corpus_embeddings(args, dataloader)

    questions_idx = [example['question_idx'] for example in dataloader]
    question_list = [example['question'] for example in dataloader]
    if args.answers_are_available:
        rationale_list = [example['rationale'] for example in dataloader]
        final_answer_list = [example['final_answer'] for example in dataloader]
    
    clustering_model = KMeans(n_clusters=args.nr_demos, random_state=args.random_seed)
    clustering_model.fit(corpus_embeddings)
    cluster_assignment = clustering_model.labels_

    clustered_sentences = [[] for i in range(args.nr_demos)]
    dist = clustering_model.transform(corpus_embeddings)
    clustered_dists = [[] for i in range(args.nr_demos)]
    clustered_idx = [[] for i in range(args.nr_demos)]
    #for sentence_id, cluster_id in enumerate(cluster_assignment):
    for sentence_id, cluster_id in zip(questions_idx, cluster_assignment):
        clustered_sentences[cluster_id].append(question_list[sentence_id])
        clustered_dists[cluster_id].append(dist[sentence_id][cluster_id])
        clustered_idx[cluster_id].append(sentence_id)

    for i in range(len(clustered_dists)):
        tmp = list(map(list, zip(range(len(clustered_dists[i])), clustered_dists[i])))
        top_min_dist = sorted(tmp, key=lambda x: x[1], reverse=False)
        if not args.sampling == "center":
            random.shuffle(top_min_dist)
        
        print(f'Cluster {i} has {len(clustered_dists[i])} elements.\n')
        demo_found=False
        cluster_demos = []
        for element in top_min_dist:
            min_idx = element[0]
            if args.answers_are_available:
                rationale = rationale_list[clustered_idx[i][min_idx]].strip()
                final_answer = final_answer_list[clustered_idx[i][min_idx]].strip()
                nr_reasoning_steps = len(rationale.replace("\n\n", "\n").split("\n"))
                if args.dataset == 'aqua':
                    nr_reasoning_steps -= 1

                question = question_list[clustered_idx[i][min_idx]] 
                if len(question.strip().split()) <= args.max_token_len \
                    and nr_reasoning_steps <= args.max_ra_len and final_answer != "":
                    demo_element = {
                        "question_idx": clustered_idx[i][min_idx],
                        "question": question,
                        "rationale": rationale,
                        "final_answer": final_answer               
                        }

                    cluster_demos.append(demo_element)
                    demo_found=True
            else:
                if len(question.strip().split()) <= args.max_ra_len:
                    demo_element = {
                        "question_idx": clustered_idx[i][min_idx],
                        "question": question,               
                        }
                    cluster_demos.append(demo_element)
                    demo_found=True
                    
            if len(cluster_demos) == args.nr_demos:
                demos = {"demo": cluster_demos}
                with open(f"{args.demos_save_dir}demos/cluster_{i}/demos", 'w', encoding="utf-8") as write_f:
                    json.dump(demos, write_f, indent=4, ensure_ascii=False)
                break
            
        if not demo_found:
            print(f'No demo found for cluster {i} since no example satisfied the constraints.\n')
            
    end = time.time()
    args_dict["execution_time"] = str(end - start) + ' seconds'

    with open(args.args_file, 'w') as f:
        json.dump(args_dict, f, indent=4)

    print('Auto demo generation for each cluster finished!')

if __name__ == "__main__":
    main()