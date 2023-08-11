from sklearn.cluster import KMeans
import json
import argparse
from langchain.embeddings import OpenAIEmbeddings
import os
from constant_vars import *
import load_env_vars
import datetime
import pickle
import time
from utils.load_data import create_dataloader
from utils.prompts_llm import build_prompt_initialize_llmchain
from utils.uncertainty_estimation import generate_uncertainty_all_questions
from utils.embedding_generation import generate_corpus_embeddings

def parse_arguments():
    parser = argparse.ArgumentParser(description="Auto-Active-CoT-KMeans")
    parser.add_argument(
        "--dataset", type=str, default="gsm8k",
        choices=["aqua", "gsm8k", "commonsensqa", "addsub", "multiarith", "strategyqa", "svamp", "singleeq", "coin_flip", "last_letters"], help="dataset used for experiment"
    )

    parser.add_argument(
        "--data_path", type=str, default="../datasets/gsm8k/train.jsonl",
        choices=["../datasets/gsm8k/train.jsonl", "../datasets/AQuA/train.json"], help="dataset used for experiment"
    )

    parser.add_argument(
        "--model_id", type=str, default="text-davinci-003", choices=["text-davinci-003", "tiiuae/falcon-7b-instruct"], help="model used for decoding."
    )

    parser.add_argument(
        "--max_ra_len", type=int, default=5, help="maximum number of reasoning chains"
    )
    parser.add_argument("--random_seed", type=int, default=42, help="random seed")

    parser.add_argument(
        "--num_trails", type=int, default=4, help="number of trails to run for each qeestion"
    )

    parser.add_argument(
        "--method", type=str, default="few_shot_cot", choices=["zero_shot_cot", "few_shot_cot"], help="method"
    )
    parser.add_argument(
        "--dataset_size_limit", type=int, default=20, help="limit the size of training data used to select the demonstrations"
    )
    parser.add_argument(
        "--sort_by", type=str, default='entropy', choices=['disagreement', 'variance', 'entropy'], help="sort the final result by given option"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7, help="temperature for llm decoding"
    )
    parser.add_argument(
        "--dir_prompts", type=str, default="uncertainty_estimation_prompts/gsm8k", help="prompts to use"
    )
    parser.add_argument(
        "--nr_demos", type=int, default=3, help='number of demonstrations'
    )

    parser.add_argument(
        "--answers_are_available", type=bool, default=True, help='true if answers are available in the test dataset, false otherwise'
    )

    # use the unsorted uncertainty file to select the demonstrations for Auto-Active-KMeans CoT
    parser.add_argument(
        "--load_uncertainty_file", type=str, default='all_uncertainties/gsm8k/unsorted_all_uncertainty_records', help='nr of demonstrations to select'
    )
    
    # gsm8k embeddings: 'embeddings/gsm8k/2023_08_10_11_45_01/embeddings.pkl'
    parser.add_argument(
        "--load_embeddings_file", type=str, default=None, help='file to load embeddings from'
    )

    args = parser.parse_args()
    if args.dataset == "gsm8k":
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"

    elif args.dataset == "aqua":
        args.direct_answer_trigger = "\nThe answer is"
    else:
        raise NotImplementedError

    if args.answers_are_available:
        args.demos_save_dir = "labeled_demos"
    else:
        args.max_ra_len = 'None'
        args.demos_save_dir = "unlabeled_demos"
        
    # "Therefore, the answer ..." -> "The answer ..."
    trigger = args.direct_answer_trigger.replace("\nTherefore, ", "")
    args.direct_answer_trigger_for_zeroshot = trigger[0].upper() + trigger[1:]
    args.direct_answer_trigger_for_zeroshot_cot = args.direct_answer_trigger
    args.direct_answer_trigger_for_fewshot = "The answer is"
    args.cot_trigger = "Let's think step by step."

    return args


def main():
    args = parse_arguments()

    current_time = datetime.datetime.now()
    time_string = current_time.strftime("%Y_%m_%d_%H_%M_%S")
    if not os.path.exists(args.demos_save_dir):
        os.makedirs(args.demos_save_dir)
        os.makedirs(args.demos_save_dir + '/' + 'auto_active_kmeans')
        os.makedirs(args.demos_save_dir + '/' +  'auto_active_kmeans' + '/' + time_string)
        os.makedirs(args.demos_save_dir + '/' + 'auto_active_kmeans' + '/' + time_string + '/' + 'demos')
        os.makedirs(args.demos_save_dir + '/' + 'auto_active_kmeans' + '/' + time_string + '/' + 'uncertainty_scores')
    elif not os.path.exists(args.demos_save_dir + '/' + 'auto_active_kmeans'):
        os.makedirs(args.demos_save_dir + '/' + 'auto_active_kmeans')
        os.makedirs(args.demos_save_dir + '/' +  'auto_active_kmeans' + '/' + time_string)
        os.makedirs(args.demos_save_dir + '/' + 'auto_active_kmeans' + '/' + time_string + '/' + 'demos')
        os.makedirs(args.demos_save_dir + '/' + 'auto_active_kmeans' + '/' + time_string + '/' + 'uncertainty_scores')
    else:
        os.makedirs(args.demos_save_dir + '/' +  'auto_active_kmeans' + '/' + time_string)
        os.makedirs(args.demos_save_dir + '/' + 'auto_active_kmeans' + '/' + time_string + '/' + 'demos')
        os.makedirs(args.demos_save_dir + '/' + 'auto_active_kmeans' + '/' + time_string + '/' + 'uncertainty_scores')

    args.args_file = args.demos_save_dir + '/' + 'auto_active_kmeans' + '/' + time_string + '/' + 'args.json'
    args.uncertainty_scores_dir = args.demos_save_dir + '/' + 'auto_active_kmeans' + '/' + time_string + '/' + 'uncertainty_scores/'
    args.demos_save_dir = args.demos_save_dir + '/' + 'auto_active_kmeans' + '/' + time_string + '/' + 'demos/'

    dataloader = create_dataloader(args)

    start = time.time()
    
    if args.load_embeddings_file:
        with open(args.load_embeddings_file, 'rb') as read_f:
            corpus_embeddings = pickle.load(read_f)
    else:
        corpus_embeddings = generate_corpus_embeddings(args, dataloader)

    if args.load_uncertainty_file: 
        with open(args.load_uncertainty_file, 'r', encoding="utf-8") as f:
            unsorted_all_uncertainty_records = json.load(f)['result']
    else:
        build_prompt_initialize_llmchain(args)


    clustering_model = KMeans(n_clusters=args.nr_demos, random_state=args.random_seed)
    clustering_model.fit(corpus_embeddings)
    cluster_assignments = clustering_model.labels_

    cluster_to_examples = [[] for i in range(args.nr_demos)]
    for example, cluster_id in zip(dataloader, cluster_assignments):
        cluster_to_examples[cluster_id].append(example)
    

    cluster_uncertainty_records_dic = {}
    demos = []
    for cluster_id in range(args.nr_demos):
        print('\n' + '*' * 50 + '\n')
        print(f'Cluster {cluster_id} has {len(cluster_to_examples[cluster_id])} examples.\n')
        cluster_examples_filtered = []
        cluster_examples = cluster_to_examples[cluster_id]

        for example in cluster_examples:
            question_idx = example['idx']
            question = example['question']
            if args.answers_are_available:
                rationale = example['rationale']
                final_answer = example['final_answer']

                if len(question.strip().split()) <= 60 and len(rationale.replace("\n\n", "\n").split("\n")) <= args.max_ra_len and final_answer != "":
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
        
        filtered_cluster_question_idxs = [example['question_idx'] for example in cluster_examples_filtered]
        print(f'After filtering out, Cluster {cluster_id} has {len(cluster_examples_filtered)} examples. These are examples idxs: {filtered_cluster_question_idxs}\n')
        if len(cluster_examples_filtered) > 0:
            if args.load_uncertainty_file:
                filtered_cluster_sorted_uncertainty_records = unsorted_all_uncertainty_records[filtered_cluster_question_idxs]
                filtered_cluster_sorted_uncertainty_records.sort(key=lambda x: -x['entropy'])
            else:
                filtered_cluster_sorted_uncertainty_records = generate_uncertainty_all_questions(args, cluster_examples_filtered)

            demos.append(filtered_cluster_sorted_uncertainty_records[0])
            cluster_uncertainty_records_dic[f'cluster_{cluster_id}'] = filtered_cluster_sorted_uncertainty_records
            print(f'Highest uncertainty example:\n{filtered_cluster_sorted_uncertainty_records[0]} \n')
        else:
            print(f'After filtering out no examples left for cluster {cluster_id+1}.\n')

    end = time.time()
    args_dict = {
        "sampling_method": "Auto_Active_KMeans",
        "dataset": args.dataset,
        "data_path": args.data_path,
        "dataset_size_limit": args.dataset_size_limit,
        "model_id": args.model_id,
        "max_ra_len": args.max_ra_len,
        "random_seed": args.random_seed,
        "num_trails": args.num_trails,
        "method": args.method,
        "sort_by": args.sort_by,
        "temperature": args.temperature,
        "dir_prompts": args.dir_prompts,
        "nr_demos": args.nr_demos,
        "answers_are_available": args.answers_are_available,
        "load_uncertainty_file": args.load_uncertainty_file,
        "load_embeddings_file": args.load_embeddings_file,
        "uncertainty_scores_dir": args.uncertainty_scores_dir,
        "demos_save_dir": args.demos_save_dir,
        "execution_time": end - start,
    }

    with open(args.args_file, 'w') as f:
        json.dump(args_dict, f, indent=4)

    demos = {"demo": demos}
    with open(args.demos_save_dir + 'demos', 'w', encoding="utf-8") as write_f:
        json.dump(demos, write_f, indent=4, ensure_ascii=False)

    with open(args.uncertainty_scores_dir + 'uncertainties', 'w', encoding="utf-8") as write_f:
        json.dump(cluster_uncertainty_records_dic, write_f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    main()