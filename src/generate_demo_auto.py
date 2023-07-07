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

def parse_arguments():
    parser = argparse.ArgumentParser(description="Auto-CoT")
    parser.add_argument(
        "--dataset", type=str, default="aqua",
        choices=["aqua", "gsm8k", "commonsensqa", "addsub", "multiarith", "strategyqa", "svamp", "singleeq", "coin_flip", "last_letters"], help="dataset used for experiment"
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
        "--demos_save_dir", type=str, default="demos/", help="directory to save the generated demos"
    )

    parser.add_argument(
        "--dataset_size_limit", type=int, default=50, help="maximum number of samples to use for clustering"
    )
    
    args = parser.parse_args()

    if args.dataset == "gsm8k":
        args.training_data_path = "../datasets/gsm8k/train.jsonl"
    elif args.dataset == "aqua":
        args.training_data_path = "../datasets/AQuA/train.json" 

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
        os.makedirs(args.demos_save_dir + 'auto_cout/' + args.dataset)

    args.demos_save_dir = f"{args.demos_save_dir}/auto_cot/{args.dataset}/"

    fix_seed(args.random_seed)

    with open(args.training_data_path) as fh:
        train_data = [json.loads(line) for line in fh.readlines() if line]
    
    if args.dataset_size_limit <= 0:
        args.dataset_size_limit = len(train_data)
    train_data = random.sample(train_data, args.dataset_size_limit)

    #encoder = SentenceTransformer(args.encoder)
    encoder = OpenAIEmbeddings()

    dataset_name = args.dataset
    max_ra_len = args.max_ra_len
    if dataset_name == "last_letters":
        max_ra_len = 7
    if dataset_name == "aqua" or dataset_name == "last_letters":
        num_clusters = 4
    elif dataset_name == "commonsensqa":
        num_clusters = 7
    elif dataset_name == "strategyqa":
        num_clusters = 6
    else:
        num_clusters = 8

    num_clusters = 3

    corpus = []
    question_list = []
    rationale_list = []
    final_answer_list = []
    
    if args.dataset == 'gsm8k':
        for example in train_data:
            question = f"Q: {example['question'].strip()}\nA:"
            corpus.append(question)
            question_list.append(question)
            rationale_list.append(f"Let's think step by step.\n'{example['answer'].split('####')[0].strip()}")
            final_answer_list.append(example["answer"].split("#### ")[-1].replace(",", ""))

    elif args.dataset == "aqua":
        for example in train_data:
            choices_str =  ' '.join([option for option in example['options']])
            question = f"Q: {example['question'].strip()} Answer Choices: {choices_str}\nA:"
            corpus.append(question)
            question_list.append(question)
            rationale_list.append(f"Let's think step by step.\n{example['rationale']}")
            final_answer_list.append(example['correct'])
    
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
            rationale = rationale_list[clustered_idx[i][min_idx]].strip()
            final_answer = final_answer_list[clustered_idx[i][min_idx]].strip()
            # print(f'R Last Char: {rationale[-1]}\n')
            # print(f'Final Answer: {final_answer}\n\n')

            # if len(question_list[clustered_idx[i][min_idx]].strip().split()) <= 60 \
            #     and len(rationale.replace("\n\n", "\n").split("\n")) <= max_ra_len and rationale[-1] == "." and final_answer != "":

                # if args.task in ["gsm8k", "multiarith", "singleeq", "addsub", "svamp"]:
                #     if not (c_pred_ans.strip() in c_rationale.split(".")[-2] or c_pred_ans.strip() in c_rationale.split()[-10:]):
                #         continue

            if len(question_list[clustered_idx[i][min_idx]].strip().split()) <= 60 \
                  and len(rationale.replace("\n\n", "\n").split("\n")) <= max_ra_len and final_answer != "":
                question = question_list[clustered_idx[i][min_idx]]
                rationale = rationale.replace("\n\n", "\n").replace("\n", " ").strip()
                rationale = " ".join(rationale.split())
                
                demo_element = {
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

    demos = {"demo": demos}
    with open(args.demos_save_dir + 'demos', 'w', encoding="utf-8") as write_f:
        json.dump(demos, write_f, indent=4, ensure_ascii=False)

    nr_reasoning_steps = [len(rat.strip().replace("\n\n", "\n").replace("\n", " ").strip().split('. ')) -1 for rat in rationale_list]

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

    plt.figure()
    plt.hist(nr_reasoning_steps, bins=5)
    plt.savefig(f"demos_plots/{args.dataset}_nr_reasoning_steps.png", dpi=600)
    plt.close()

if __name__ == "__main__":
    main()