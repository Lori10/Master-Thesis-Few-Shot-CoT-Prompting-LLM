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
from generate_demo_active import create_uncertainty

def parse_arguments():
    parser = argparse.ArgumentParser(description="Auto-Active-CoT-Combination")
    parser.add_argument(
        "--dataset", type=str, default="gsm8k",
        choices=["aqua", "gsm8k", "commonsensqa", "addsub", "multiarith", "strategyqa", "svamp", "singleeq", "coin_flip", "last_letters"], help="dataset used for experiment"
    )
    parser.add_argument(
        "--model", type=str, default="text-davinci-002", choices=["text-davinci-002", "code-davinci-002"], help="model used for decoding."
    )
    parser.add_argument(
        "--max_ra_len", type=int, default=5, help="maximum number of reasoning chains"
    )
    parser.add_argument("--random_seed", type=int, default=42, help="random seed")
    parser.add_argument(
        "--num_trails", type=int, default=5, help="number of trails to run for each qeestion"
    )
    parser.add_argument(
        "--sampling", type=str, default="center", help="whether to sample the cluster center first"
    )
    parser.add_argument(
        "--demos_save_dir", type=str, default="demos/", help="directory to save the generated demos"
    )

    parser.add_argument(
        "--method", type=str, default="few_shot_cot", choices=["zero_shot_cot", "few_shot_cot"], help="method"
    )
    parser.add_argument(
        "--qes_limit", type=int, default=10, help="maximum number of samples to use for clustering"
    )
    parser.add_argument(
        "--sort_by", type=str, default='disagreement', choices=['disagreement', 'variance', 'entropy'], help="sort the final result by given option"
    )
    parser.add_argument(
        "--max_length_cot", type=int, default=256, help="maximum length of output tokens by model for reasoning extraction"
    )
    parser.add_argument(
        "--api_time_interval", type=float, default=1.0, help="how many seconds sleep between each request"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7, help=""
    )
    parser.add_argument(
        "--dir_prompts", type=str, default="prompts_active", help="prompts to use"
    )
    parser.add_argument(
        "--concat_length", type=int, default=2, help='Used for task last_letters, indicates length of last letter concat'
    )

    args = parser.parse_args()

    if args.dataset == "gsm8k":
        args.data_path = "../datasets/gsm8k/train.jsonl"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"

    elif args.dataset == "aqua":
        args.data_path = "../datasets/AQuA/train.json" 
        args.direct_answer_trigger = "\nThe answer is"

    # "Therefore, the answer ..." -> "The answer ..."
    trigger = args.direct_answer_trigger.replace("\nTherefore, ", "")
    args.direct_answer_trigger_for_zeroshot = trigger[0].upper() + trigger[1:]
    args.direct_answer_trigger_for_zeroshot_cot = args.direct_answer_trigger
    args.direct_answer_trigger_for_fewshot = "The answer is"
    args.cot_trigger = "Let's think step by step."

    return args


def main():
    args = parse_arguments()
    if not os.path.exists(args.demos_save_dir):
        os.makedirs(args.demos_save_dir)
        os.makedirs(args.demos_save_dir + 'auto_active_cot')
        os.makedirs(args.demos_save_dir + 'auto_active_cot/' + args.dataset)
    elif not os.path.exists(args.demos_save_dir + 'auto_active_cot'):
        os.makedirs(args.demos_save_dir + 'auto_active_cot')
        os.makedirs(args.demos_save_dir + 'auto_active_cot/' + args.dataset)
    elif not os.path.exists(args.demos_save_dir + 'auto_active_cot/' + args.dataset):
        os.makedirs(args.demos_save_dir + 'auto_active_cout/' + args.dataset)

    uncertainty_estimation_dir = f"{args.demos_save_dir}auto_active_cot/uncertainty_estimation/"
    if not os.path.exists(uncertainty_estimation_dir):
        os.makedirs(uncertainty_estimation_dir)

    args.demos_save_dir = f"{args.demos_save_dir}auto_active_cot/{args.dataset}/"

    set_random_seed(args.random_seed)
    dataloader = create_dataloader(args)

    if args.dataset_size > args.qes_limit:
        dataloader = dataloader[:args.qes_limit] # replace 7 with 1000; only take 1000 questions randomly to annotate, randomness decided by seed
    print(f"Dataloader size: {len(dataloader)}")

    if args.qes_limit == 0:
        args.qes_limit = len(dataloader)


    corpus = [example['question'] for example in dataloader]
    question_list = [example['question'] for example in dataloader]
    rationale_list = [example['rationale'] for example in dataloader]
    final_answer_list = [example['final_answer'] for example in dataloader] 

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
    encoder = OpenAIEmbeddings()

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
    cluster_uncertainty_records = {}

    for i in range(len(clustered_dists)):
        print("Cluster ", i+1)
        tmp = list(map(list, zip(range(len(clustered_dists[i])), clustered_dists[i])))
        top_min_dist = sorted(tmp, key=lambda x: x[1], reverse=False)
        if not args.sampling == "center":
            random.shuffle(top_min_dist)

        cluster_selected_demos = []
        
        for element in top_min_dist:
            min_idx = element[0]
            rationale = rationale_list[clustered_idx[i][min_idx]].strip()
            final_answer = final_answer_list[clustered_idx[i][min_idx]].strip()

            if len(question_list[clustered_idx[i][min_idx]].strip().split()) <= 60 \
                  and len(rationale.replace("\n\n", "\n").split("\n")) <= max_ra_len and final_answer != "":
                question = question_list[clustered_idx[i][min_idx]]
                rationale = rationale.replace("\n\n", "\n").replace("\n", " ").strip()
                rationale = " ".join(rationale.split())
                
                demo_element = {
                    "question_idx": clustered_idx[i][min_idx],
                    "question": question,
                    "rationale": rationale,
                    "final_answer": final_answer,
                    }
                cluster_selected_demos.append(demo_element)

        print(f'Cluster {i+1} has {len(cluster_selected_demos)} demos based on nr of reasoning steps and question length\n\n')
        result = create_uncertainty(args, cluster_selected_demos)
        cluster_uncertainty_records[f"cluster_{i}"] = result

        if args.sort_by == "disagreement":
            if args.dataset == "strategyqa":
                try:
                    # sort based on the entropy or the difference between yes and no answers
                    result.sort(key=lambda x: abs(x['occurrence']['yes'] - x['occurrence']['no']))
                except:
                    # sort by disagreement
                    result.sort(key=lambda x: -len(x['occurrence']))
            else:
                result.sort(key=lambda x: -len(x['occurrence']))
        elif args.sort_by == "variance" and args.dataset in ("gsm8k", "asdiv", "svamp", "singleeq", "addsub", "multiarith"):
            # sort by variance
            result.sort(key=lambda x: -x['variance'])
        elif args.sort_by == "entropy" :
            result.sort(key=lambda x:-x['entropy'])

        print(f'Demo with highest uncertainty from cluster {i+1}:\n')
        print(result[0])
        print('\n')
        print('*' * 70)
        print('\n')
        demos.append(result[0])


    demos = {"demo": demos}
    with open(args.demos_save_dir + 'demos', 'w', encoding="utf-8") as write_f:
        json.dump(demos, write_f, indent=4, ensure_ascii=False)

    with open(f"{uncertainty_estimation_dir}{args.dataset}_k_{args.num_trails}", 'w', encoding="utf-8") as write_f:
        json.dump(cluster_uncertainty_records, write_f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    main()