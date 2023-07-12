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
    parser = argparse.ArgumentParser(description="Auto-Active-CoT-Combination-KMeans")
    parser.add_argument(
        "--dataset", type=str, default="gsm8k",
        choices=["aqua", "gsm8k", "commonsensqa", "addsub", "multiarith", "strategyqa", "svamp", "singleeq", "coin_flip", "last_letters"], help="dataset used for experiment"
    )

    parser.add_argument(
        "--data_path", type=str, default="../datasets/gsm8k/train.jsonl",
        choices=["../datasets/gsm8k/train.jsonl", "../datasets/AQuA/train.json"], help="dataset used for experiment"
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
        "--dataset_size_limit", type=int, default=5, help="limit the size of training data used to select the demonstrations"
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
    parser.add_argument(
        "--nr_demos", type=int, default=3, help='number of demonstrations'
    )

    args = parser.parse_args()

    if args.dataset == "gsm8k":
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"

    elif args.dataset == "aqua":
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

    random.seed(args.random_seed)
    dataloader = create_dataloader(args, answers_available=True)

    if args.dataset_size_limit <= 0:
        args.dataset_size_limit = len(dataloader)
    else:
        dataloader = dataloader[:args.dataset_size_limit] # replace 7 with 1000; only take 1000 questions randomly to annotate, randomness decided by seed
    print(f"Dataloader size: {len(dataloader)}")

    corpus = [example['question'] for example in dataloader]
    question_list = [example['question'] for example in dataloader]
    rationale_list = [example['rationale'] for example in dataloader]
    final_answer_list = [example['final_answer'] for example in dataloader] 

    max_ra_len = args.max_ra_len
    num_clusters = args.nr_demos
    encoder = OpenAIEmbeddings()

    corpus_embeddings = np.array(encoder.embed_documents(corpus))
    clustering_model = KMeans(n_clusters=num_clusters, random_state=args.random_seed)
    clustering_model.fit(corpus_embeddings)
    cluster_assignments = clustering_model.labels_

    cluster_to_examples = [[] for i in range(num_clusters)]
    question_idxs = list(range(len(question_list)))
    for question_idx, question, rationale, final_answer, cluster_id in zip(question_idxs, question_list, rationale_list, final_answer_list, cluster_assignments):
        cluster_to_examples[cluster_id].append((question, rationale, final_answer))
        
    cluster_uncertainty_records = {}
    demos = []
    for cluster_id in range(num_clusters):
        print(f'Cluster {cluster_id+1} has {len(cluster_to_examples[cluster_id])} examples.\n')

        cluster_selected_demos = []
        cluster_examples = cluster_to_examples[cluster_id]
        for example in cluster_examples:
            question = example[0]
            rationale = example[1]
            final_answer = example[2]

            if len(question.strip().split()) <= 60 and len(rationale.replace("\n\n", "\n").split("\n")) <= max_ra_len and final_answer != "":
                rationale = rationale.replace("\n\n", "\n").replace("\n", " ").strip()
                rationale = " ".join(rationale.split())
                
                demo_element = {
                    "question_idx" : question_idx,
                    "question": question,
                    "rationale": rationale,
                    "final_answer": final_answer,
                    }
                cluster_selected_demos.append(demo_element)

        print(f'Cluster {cluster_id+1} has {len(cluster_selected_demos)} demos after selecting examples based on nr of reasoning steps and question length\n\n')
        result = create_uncertainty(args, cluster_selected_demos)
        cluster_uncertainty_records[f"cluster_{cluster_id}"] = result
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

        print(f'Demo with highest uncertainty from cluster {cluster_id+1}:\n')
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