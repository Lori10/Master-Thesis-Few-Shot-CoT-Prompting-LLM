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
        "--model", type=str, default="gpt-3.5-turbo", choices=["gpt-3.5-turbo"], help="model used for decoding."
    )
    parser.add_argument(
        "--max_ra_len", type=int, default=5, help="maximum number of reasoning chains"
    )
    parser.add_argument("--random_seed", type=int, default=42, help="random seed")
    parser.add_argument(
        "--num_trails", type=int, default=5, help="number of trails to run for each qeestion"
    )

    parser.add_argument(
        "--method", type=str, default="few_shot_cot", choices=["zero_shot_cot", "few_shot_cot"], help="method"
    )
    parser.add_argument(
        "--dataset_size_limit", type=int, default=10, help="limit the size of training data used to select the demonstrations"
    )
    parser.add_argument(
        "--sort_by", type=str, default='entropy', choices=['disagreement', 'variance', 'entropy'], help="sort the final result by given option"
    )
    # parser.add_argument(
    #     "--max_length_cot", type=int, default=256, help="maximum length of output tokens by model for reasoning extraction"
    # )
    # parser.add_argument(
    #     "--api_time_interval", type=float, default=1.0, help="how many seconds sleep between each request"
    # )
    parser.add_argument(
        "--temperature", type=float, default=0.7, help="temperature for llm decoding"
    )
    parser.add_argument(
        "--dir_prompts", type=str, default="prompts_active", help="prompts to use"
    )
    parser.add_argument(
        "--nr_demos", type=int, default=3, help='number of demonstrations'
    )

    parser.add_argument(
        "--uncertainty_per_cluster_dir", type=str, default='uncertainty_scores/', help='directory where the uncertainty scores are saved'
    )

    parser.add_argument(
        "--answers_are_available", type=bool, default=False, help='true if answers are available in the test dataset, false otherwise'
    )

    args = parser.parse_args()

    if args.dataset == "gsm8k":
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"

    elif args.dataset == "aqua":
        args.direct_answer_trigger = "\nThe answer is"
    else:
        raise NotImplementedError

    if args.answers_are_available:
        args.demos_save_dir = "labeled_demos/"
    else:
        args.demos_save_dir = "unlabeled_demos/"
        
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
        os.makedirs(args.demos_save_dir + 'auto_active_cot_kmeans')
        os.makedirs(args.demos_save_dir + 'auto_active_cot_kmeans/' + args.dataset)
    elif not os.path.exists(args.demos_save_dir + 'auto_active_cot_kmeans'):
        os.makedirs(args.demos_save_dir + 'auto_active_cot_kmeans')
        os.makedirs(args.demos_save_dir + 'auto_active_cot_kmeans/' + args.dataset)
    elif not os.path.exists(args.demos_save_dir + 'auto_active_cot_kmeans/' + args.dataset):
        os.makedirs(args.demos_save_dir + 'auto_active_cot_kmeans/' + args.dataset)

    args.demos_save_dir = f"{args.demos_save_dir}auto_active_cot_kmeans/{args.dataset}/"

    if not os.path.exists(args.uncertainty_per_cluster_dir):
        os.makedirs(args.uncertainty_per_cluster_dir)
    uncertainty_filepath = f"{args.uncertainty_per_cluster_dir}AutoActiveKMeans_{args.dataset}_numtrials_{args.num_trails}_nrclusters_{args.nr_demos}_sortby_{args.sort_by}"

    random.seed(args.random_seed)
    dataloader = create_dataloader(args)

    if args.dataset_size_limit <= 0:
        args.dataset_size_limit = len(dataloader)
    else:
        dataloader = dataloader[:args.dataset_size_limit] # replace 7 with 1000; only take 1000 questions randomly to annotate, randomness decided by seed
    print(f"Proceeding with data size: {len(dataloader)}")

    corpus = [example['question'] for example in dataloader]
    question_list = [example['question'] for example in dataloader]
    if args.answers_are_available:
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
    if args.answers_are_available:
        for question_idx, question, rationale, final_answer, cluster_id in zip(question_idxs, question_list, rationale_list, final_answer_list, cluster_assignments):
            cluster_to_examples[cluster_id].append({'question_idx': question_idx, 
                                                    'question' : question,
                                                    'rationale' : rationale,
                                                    'final_answer': final_answer
            })
    else:
        for question_idx, question, cluster_id in zip(question_idxs, question_list, cluster_assignments):
            cluster_to_examples[cluster_id].append({'question_idx': question_idx, 
                                                    'question' : question,
            })
        
    cluster_uncertainty_records_dic = {}
    demos = []
    for cluster_id in range(num_clusters):
        print(f'Cluster {cluster_id+1} has {len(cluster_to_examples[cluster_id])} examples.\n')

        cluster_examples_filtered = []
        cluster_examples = cluster_to_examples[cluster_id]

        for example in cluster_examples:
            question_idx = example['question_idx']
            question = example['question']
            if args.answers_are_available:
                rationale = example['rationale']
                final_answer = example['final_answer']

                if len(question.strip().split()) <= 60 and len(rationale.replace("\n\n", "\n").split("\n")) <= max_ra_len and final_answer != "":
                    rationale = rationale.replace("\n\n", "\n").replace("\n", " ").strip()
                    rationale = " ".join(rationale.split())
                    
                    demo_element = {
                        "question_idx" : question_idx,
                        "question": question,
                        "rationale": rationale,
                        "final_answer": final_answer,
                        }
                    cluster_examples_filtered.append(demo_element)
                    
            else:
                if len(question.strip().split()) <= 60:        
                    cluster_examples_filtered.append(example)
        
        print(f'After filtering out, Cluster {cluster_id+1} has {len(cluster_examples_filtered)} examples.\n')
        if len(cluster_examples_filtered) > 0:
            cluster_uncertainty_records = create_uncertainty(args, cluster_examples_filtered)  
            print(f'Highest uncertainty example:\n{cluster_uncertainty_records[0]}')                   
            demos.append(cluster_uncertainty_records[0])
            cluster_uncertainty_records_dic[f'cluster_{cluster_id}'] = cluster_uncertainty_records
        else:
            print(f'After filtering out no examples left for cluster {cluster_id+1}.\n')


    demos = {"demo": demos}
    with open(args.demos_save_dir + 'demos', 'w', encoding="utf-8") as write_f:
        json.dump(demos, write_f, indent=4, ensure_ascii=False)

    with open(uncertainty_filepath, 'w', encoding="utf-8") as write_f:
        json.dump(cluster_uncertainty_records_dic, write_f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    main()