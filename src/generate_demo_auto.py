import random
# from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np
import json
import matplotlib.pyplot as plt
import argparse
from utils import fix_seed
from langchain.embeddings import OpenAIEmbeddings
import os
from utils import *

def parse_arguments():
    parser = argparse.ArgumentParser(description="Auto-CoT")
    parser.add_argument(
        "--dataset", type=str, default="gsm8k",
        choices=["aqua", "gsm8k", "commonsensqa", "addsub", "multiarith", "strategyqa", "svamp", "singleeq", "coin_flip", "last_letters"], help="dataset used for experiment"
    )

    parser.add_argument(
        "--data_path", type=str, default="../datasets/gsm8k/train.jsonl",
        choices=["../datasets/gsm8k/train.jsonl", "../datasets/AQuA/train.json"], help="dataset used for experiment"
    )

    parser.add_argument(
        "--max_ra_len", type=int, default=5, help="maximum number of reasoning chains"
    )
    parser.add_argument("--random_seed", type=int, default=42, help="random seed")
    # parser.add_argument(
    #     "--encoder", type=str, default="all-MiniLM-L6-v2", help="which sentence-transformer encoder for clustering"
    # )
    parser.add_argument(
        "--sampling", type=str, default="center", help="whether to sample the cluster center first"
    )

    parser.add_argument(
        "--dataset_size_limit", type=int, default=50, help="whether to limit training dataset size. if 0, the dataset size is unlimited and we use all the samples in the dataset for creating the demonstrations."
    )

    parser.add_argument(
        "--nr_demos", type=int, default=6, help="nr of demonstrations to select"
    )
    
    parser.add_argument(
        "--answers_are_available", type=bool, default=False, help='true if answers are available in the test dataset, false otherwise'
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
        os.makedirs(args.demos_save_dir + 'auto_cot')
        os.makedirs(args.demos_save_dir + 'auto_cot/' + args.dataset)
    elif not os.path.exists(args.demos_save_dir + 'auto_cot'):
        os.makedirs(args.demos_save_dir + 'auto_cot')
        os.makedirs(args.demos_save_dir + 'auto_cot/' + args.dataset)
    elif not os.path.exists(args.demos_save_dir + 'auto_cot/' + args.dataset):
        os.makedirs(args.demos_save_dir + 'auto_cot/' + args.dataset)

    args.demos_save_dir = f"{args.demos_save_dir}auto_cot/{args.dataset}/"


    random.seed(args.random_seed)
    dataloader = create_dataloader(args)

    if args.dataset_size_limit <= 0:
        args.dataset_size_limit = len(dataloader)
    else:
        dataloader = dataloader[:args.dataset_size_limit] # replace 7 with 1000; only take 1000 questions randomly to annotate, randomness decided by seed
    print(f"Dataloader size: {len(dataloader)}")

    #encoder = SentenceTransformer(args.encoder)
    encoder = OpenAIEmbeddings()

    max_ra_len = args.max_ra_len
    num_clusters = args.nr_demos

    question_idxs = [example['question_idx'] for example in dataloader]
    corpus = [example['question'] for example in dataloader]
    question_list = [example['question'] for example in dataloader]

    if args.answers_are_available:
        rationale_list = [example['rationale'] for example in dataloader]
        final_answer_list = [example['final_answer'] for example in dataloader]
    
    corpus_embeddings = np.array(encoder.embed_documents(corpus))
    clustering_model = KMeans(n_clusters=num_clusters, random_state=args.random_seed)
    clustering_model.fit(corpus_embeddings)
    cluster_assignment = clustering_model.labels_

    clustered_sentences = [[] for i in range(num_clusters)]

    dist = clustering_model.transform(corpus_embeddings)
    clustered_dists = [[] for i in range(num_clusters)]
    clustered_idx = [[] for i in range(num_clusters)]
    for sentence_id, cluster_id in enumerate(cluster_assignment):
        clustered_sentences[cluster_id].append(corpus[sentence_id])
        clustered_dists[cluster_id].append(dist[sentence_id][cluster_id])
        clustered_idx[cluster_id].append(sentence_id)

    demos = []
    
    for i in range(len(clustered_dists)):
        print("Cluster ", i+1)
        tmp = list(map(list, zip(range(len(clustered_dists[i])), clustered_dists[i])))
        top_min_dist = sorted(tmp, key=lambda x: x[1], reverse=False)
        if not args.sampling == "center":
            random.shuffle(top_min_dist)
        for element in top_min_dist:
            min_idx = element[0]

            if args.answers_are_available:
                rationale = rationale_list[clustered_idx[i][min_idx]].strip()
                final_answer = final_answer_list[clustered_idx[i][min_idx]].strip()

                if len(question_list[clustered_idx[i][min_idx]].strip().split()) <= 60 \
                    and len(rationale.replace("\n\n", "\n").split("\n")) <= max_ra_len and final_answer != "":
                    question_idx = question_idxs[clustered_idx[i][min_idx]]
                    question = question_list[clustered_idx[i][min_idx]]
                    rationale = rationale.replace("\n\n", "\n").replace("\n", " ").strip()
                    rationale = " ".join(rationale.split())
                    
                    demo_element = {
                        "question_idx": question_idx,
                        "question": question,
                        "rationale": rationale,
                        "final_answer": final_answer               
                        }
                    demos.append(demo_element)
                    print(f'Q:\n{question}\n')
                    print(f'R:\n{rationale}\n')
                    print(f'FA:\n{final_answer}\n\n')
                    print("")
                    break
            else:
                if len(question_list[clustered_idx[i][min_idx]].strip().split()) <= 60:
                    question_idx = question_idxs[clustered_idx[i][min_idx]]
                    question = question_list[clustered_idx[i][min_idx]]        
                    demo_element = {
                        "question_idx": question_idx,
                        "question": question,               
                        }
                    demos.append(demo_element)
                    print(f'Q:\n{question}\n')
                    print("")
                    break

    demos = {"demo": demos}
    with open(args.demos_save_dir + 'demos', 'w', encoding="utf-8") as write_f:
        json.dump(demos, write_f, indent=4, ensure_ascii=False)
    

    y_km = clustering_model.fit_predict(corpus_embeddings)
    pca_model = PCA(n_components=2, random_state=args.random_seed)
    transformed = pca_model.fit_transform(corpus_embeddings)
    centers = pca_model.transform(clustering_model.cluster_centers_)
    
    plt.figure()
    plt.scatter(x=transformed[:, 0], y=transformed[:, 1], c=y_km, s=50, cmap=plt.cm.Paired, alpha=0.4)
    plt.scatter(centers[:, 0],centers[:, 1],
            s=250, marker='*', label='centroids',
            edgecolor='black',
           c=np.arange(0,num_clusters),cmap=plt.cm.Paired,)
    plt.xticks([])
    plt.yticks([])
    plt.savefig(f"demos_plots/{args.dataset}_clustering.png", dpi=600)
    plt.close()

    if args.answers_are_available:
        nr_reasoning_steps = [len(rat.strip().replace("\n\n", "\n").replace("\n", " ").strip().split('. ')) -1 for rat in rationale_list]
        plt.figure()
        plt.hist(nr_reasoning_steps, bins=5)
        plt.savefig(f"demos_plots/{args.dataset}_nr_reasoning_steps.png", dpi=600)
        plt.close()

if __name__ == "__main__":
    main()