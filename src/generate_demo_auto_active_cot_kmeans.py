import random
# from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import numpy as np
import json
import argparse
from utils import fix_seed
from langchain.embeddings import OpenAIEmbeddings
import os
from utils import *
from generate_demo_active import create_uncertainty
from constant_vars import *
import load_env_vars
import datetime
import openai
from utils import initialize_llmchain

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
        "--dataset_size_limit", type=int, default=10, help="limit the size of training data used to select the demonstrations"
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

    args.json_file = args.demos_save_dir + '/' + 'auto_active_kmeans' + '/' + time_string + '/' + 'args.json'
    args.uncertainty_scores_dir = args.demos_save_dir + '/' + 'auto_active_kmeans' + '/' + time_string + '/' + 'uncertainty_scores/'
    args.demos_save_dir = args.demos_save_dir + '/' + 'auto_active_kmeans' + '/' + time_string + '/' + 'demos/'

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
    }

    with open(args.json_file, 'w') as f:
        json.dump(args_dict, f, indent=4)


    if args.dataset == "gsm8k":
        prefix = prefix_gsm8k
    elif args.dataset == "aqua":
        prefix = prefix_aqua
    else:
        raise NotImplementedError("dataset not implemented")

    print('Hyperparameters:')
    random.seed(args.random_seed)

    if args.method == "few_shot_cot":
        args.prefix = prefix + ' Follow the format of the examples below:\n'
        given_prompt_list = create_several_input_prompts(args, cot_flag=True)
        assert len(given_prompt_list) == 1
        args.prompt = given_prompt_list[0]
    elif args.method == "zero_shot_cot":
        args.prompt = prefix + "\nQ: " + "{question}" + "\nA: Let's think step by step."

    args.llm_chain = initialize_llmchain(args.prompt, args)
    
    print('Prompt for uncertainty estimation:\n' + args.prompt)
    print('*' * 50)

    dataloader = create_dataloader(args)

    if args.dataset_size_limit <= 0:
        args.dataset_size_limit = len(dataloader)
    else:
        dataloader = dataloader[:args.dataset_size_limit] # replace 7 with 1000; only take 1000 questions randomly to annotate, randomness decided by seed
    print(f"Proceeding with data size: {len(dataloader)}")
    print('hyperparameters ony by one')
    print(f'args.data_path: {args.data_path}')
    print(f'args.model_id: {args.model_id}')
    print(f'args.max_ra_len: {args.max_ra_len}')
    print(f'args.random_seed: {args.random_seed}')
    print(f'args.num_trails: {args.num_trails}')
    print(f'args.method: {args.method}')
    print(f'args.sort_by: {args.sort_by}')
    print(f'args.temperature: {args.temperature}')
    print(f'args.dir_prompts: {args.dir_prompts}')
    print(f'args.nr_demos: {args.nr_demos}')
    print(f'args.answers_are_available: {args.answers_are_available}')
    
    corpus = [example['question'] for example in dataloader]
    question_list = [example['question'] for example in dataloader]
    if args.answers_are_available:
        rationale_list = [example['rationale'] for example in dataloader]
        final_answer_list = [example['final_answer'] for example in dataloader] 

    max_ra_len = args.max_ra_len
    num_clusters = args.nr_demos

    openai.api_key = os.getenv("OPENAI_API_KEY")
    headers = {
        "x-api-key": openai.api_key,
    }

    encoder = OpenAIEmbeddings(
        deployment="text-embedding-ada-002-v2", headers=headers, chunk_size=1
    )

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
        print(f'Cluster {cluster_id} has {len(cluster_to_examples[cluster_id])} examples.\n')

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
        
        print(f'After filtering out, Cluster {cluster_id} has {len(cluster_examples_filtered)} examples.\n')
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

    with open(args.uncertainty_scores_dir + 'uncertainties', 'w', encoding="utf-8") as write_f:
        json.dump(cluster_uncertainty_records_dic, write_f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    main()