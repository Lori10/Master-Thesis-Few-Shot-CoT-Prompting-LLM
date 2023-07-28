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
from generate_demo_active import generate_uncertainty_qes
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy import stats
import pickle
from sklearn.metrics import pairwise_distances


def parse_arguments():
    parser = argparse.ArgumentParser(description="Auto-Active-CoT-Combination-KMeansPlusPlus")
    parser.add_argument(
        "--dataset", type=str, default="gsm8k",
        choices=["aqua", "gsm8k"], help="dataset used for experiment"
    )

    parser.add_argument(
        "--data_path", type=str, default="../datasets/gsm8k/train.jsonl",
        choices=["../datasets/gsm8k/train.jsonl", "../datasets/AQuA/train.json"], help="dataset used for experiment"
    )

    parser.add_argument(
        "--model_id", type=str, default="gpt-3.5-turbo", choices=["gpt-3.5-turbo", "tiiuae/falcon-7b-instruct"], help="model used for decoding."
    )

    parser.add_argument(
        "--normalize_distance_uncertainty", type=bool, default=True, help="whether to normalize the distance uncertainty before applying F1 score"
    )
    # parser.add_argument(
    #     "--max_ra_len", type=int, default=5, help="maximum number of reasoning chains"
    # )
    parser.add_argument("--random_seed", type=int, default=42, help="random seed")
    parser.add_argument(
        "--num_trails", type=int, default=5, help="number of trails to run for each question"
    )
    parser.add_argument(
        "--sampling", type=str, default="center", help="whether to sample the cluster center first"
    )
    parser.add_argument(
        "--method", type=str, default="few_shot_cot", choices=["zero_shot_cot", "few_shot_cot"], help="method"
    )

    parser.add_argument(
        "--dir_prompts", type=str, default="uncertainty_estimation_prompts/gsm8k", help="prompts to use for uncertainty estimation"
    )
    
    parser.add_argument(
        "--dataset_size_limit", type=int, default=10, help="limit the size of training data used to select the demonstrations"
    )
    parser.add_argument(
        "--sort_by", type=str, default='entropy', choices=['disagreement', 'variance', 'entropy'], help="sort the final result by given option"
    )

    parser.add_argument(
        "--distance_metric", type=str, default='cosine', choices=['cosine', 'euclidean'], help="sort the final result by given option"
    )

    parser.add_argument(
        "--beta", type=int, default=1.7, help="weight for uncertainty. For example beta=2 means uncertainty is twice as important as the distance"
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
        "--uncertainty_scores_dir", type=str, default='uncertainty_scores/', help='directory where the uncertainty scores are saved'
    )

    parser.add_argument(
        "--nr_demos", type=int, default=9, help='number of demonstrations'
    )
    parser.add_argument(
        "--answers_are_available", type=bool, default=True, help='true if answers are available in the test dataset, false otherwise'
    )

    parser.add_argument(
        "--greedy", type=bool, default=True, help='whether to select examples with the highest f1-score or use random-weighted sampling'
    )

    parser.add_argument(
        "--auto_active_limit_nr", type=int, default=500, help='the number of examples to use for auto-active labeling'
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

def f1_score(distances, uncertainties, args):
    # distances is the precision, uncertainties is the recall, beta is the weight of recall
    distances = np.array(distances)
    uncertainties = np.array(uncertainties)
    #print(f'Distances before Normalization:\n{distances}')
    #print(f'Uncertainties before Normalization:\n{uncertainties}')
    print('---------------------------------------------')
    if args.normalize_distance_uncertainty:
        distances = distances / sum(distances)
        uncertainties = uncertainties / sum(uncertainties)
        #print(f'Distances after Normalization:\n{distances}')
        #print(f'Uncertainties after Normalization:\n{uncertainties}')
        # print(distances[3])
        # print(uncertainties[3])
        # print('------------')

    f1_scores = ((args.beta**2 + 1) * distances * uncertainties) / (args.beta**2 * distances + uncertainties)
    f1_scores[np.isnan(f1_scores)] = 0
    return f1_scores, distances, uncertainties

def square_prob(scores):
    return (scores ** 2)/ sum(scores ** 2)
    
def softmax(scores):
    return np.exp(scores) / np.sum(np.exp(scores), axis=0)

def normalization(scores):
    return (scores - min(scores)) / (max(scores) - min(scores))

def generate_doc_embedding(corpus, encoder=OpenAIEmbeddings()):
    return np.array(encoder.embed_documents(corpus))

def main():
    args = parse_arguments()
    if args.greedy:
        greedy_str = 'greedy'
    else:
        greedy_str = 'random_weighted'

    if not os.path.exists(args.demos_save_dir):
        os.makedirs(args.demos_save_dir)
        os.makedirs(args.demos_save_dir + f'auto_active_cot_kmeans_plusplus_{greedy_str}')
        os.makedirs(args.demos_save_dir + f'auto_active_cot_kmeans_plusplus_{greedy_str}/' + args.dataset)
    elif not os.path.exists(args.demos_save_dir + f'auto_active_cot_kmeans_plusplus_{greedy_str}'):
        os.makedirs(args.demos_save_dir + f'auto_active_cot_kmeans_plusplus_{greedy_str}')
        os.makedirs(args.demos_save_dir + f'auto_active_cot_kmeans_plusplus_{greedy_str}/' + args.dataset)
    elif not os.path.exists(args.demos_save_dir + f'auto_active_cot_kmeans_plusplus_{greedy_str}/' + args.dataset):
        os.makedirs(args.demos_save_dir + f'auto_active_cot_kmeans_plusplus_{greedy_str}/' + args.dataset)

    args.demos_save_dir = f"{args.demos_save_dir}auto_active_cot_kmeans_plusplus_{greedy_str}/{args.dataset}/"

    if not os.path.exists(args.uncertainty_scores_dir):
        os.makedirs(args.uncertainty_scores_dir)


    if '/' in args.model_id:
        model_name = args.model_id.replace('/', '-')   
    else:
        model_name = args.model_id
    distance_uncertainty_filepath = f"{args.uncertainty_scores_dir}method_AutoActiveKMeansPlusPlus_{greedy_str}_dataset_{args.dataset}_model_{model_name}_numtrials_{args.num_trails}_sortby_{args.sort_by}.txt"

    set_random_seed(args.random_seed)

    
    dataloader = create_dataloader(args)
    if args.dataset_size_limit <= 0:
        args.dataset_size_limit = len(dataloader)
    else:
        dataloader = dataloader[:args.dataset_size_limit] # replace 7 with 1000; only take 1000 questions randomly to annotate, randomness decided by seed
    print(f"Proceeding with data size: {len(dataloader)}")

    # if args.auto_active_limit_nr > len(dataloader):
    #     args.auto_active_limit_nr = len(dataloader)
    # if args.nr_demos > args.auto_active_limit_nr:

    #uncertainty_list = []
    corpus = []
    questions_idxs = []
    for idx, example in enumerate(dataloader):
        #print(f'Question: {example["question"]}\n')
        #uncertainty_record = generate_uncertainty_qes(args, example)
        corpus.append(example['question'])
        #uncertainty_list.append(uncertainty_record['entropy'])
        questions_idxs.append(idx)

    # with open('uncertainties.pkl', 'wb') as f:
    #     pickle.dump(uncertainty_list, f)
    

    #encoder = OpenAIEmbeddings()
    #embeddings = np.array(encoder.embed_documents(corpus))
    
    file = open("embeddings", "rb")
    embeddings = np.load(file)

    with open('uncertainties.pkl', 'rb') as f:
        uncertainty_list = pickle.load(f)

    uncertainties_series = pd.Series(data=uncertainty_list, index=questions_idxs)
    first_question_idx = list(uncertainties_series.sort_values(ascending=False).head(1).index)[0]
    selected_idxs = [first_question_idx]
    selected_data = [embeddings[first_question_idx]]
    all_info_list = [{'iteration' : 0,
                     'selected_idx': first_question_idx,
                    }] 
    j = 1
    demos = []
    print(f'Iteration: {j}')
    print(f'All indices: {questions_idxs}')
    print(f'Selected indices: {selected_idxs}')
    while j < args.nr_demos:
        if len(selected_data) == 1:
            D2 = pairwise_distances(embeddings, selected_data, metric=args.distance_metric, n_jobs=-1).ravel().astype(float)
            if args.distance_metric == 'cosine':
                D2 = 1 - D2
                
        else:
            newD = pairwise_distances(embeddings, [selected_data[-1]], metric=args.distance_metric, n_jobs=-1).ravel().astype(float)
            if args.distance_metric == 'cosine':
                newD = 1 - newD
                for i in range(len(embeddings)):
                    if D2[i] < newD[i]:
                        D2[i] = newD[i]

            elif args.distance_metric == 'euclidean':
                for i in range(len(embeddings)):
                    if D2[i] > newD[i]:
                        D2[i] = newD[i]

        # convert into exponential distances for cosine
        if args.distance_metric == 'cosine':
            D2[D2 > 0.999] = 1
            not_selected_questions_distances_uncertainties = [(question_idx, np.exp(-distance), uncertainty) for question_idx, distance, uncertainty in zip(questions_idxs, D2, uncertainty_list) if distance != 1]

        elif args.distance_metric == 'euclidean':
            D2[D2 < 0.0001] = 0
            not_selected_questions_distances_uncertainties = [(question_idx, distance, uncertainty) for question_idx, distance, uncertainty in zip(questions_idxs, D2, uncertainty_list) if distance != 0]

        not_selected_questions_idxs = [question_idx for question_idx, _, _ in not_selected_questions_distances_uncertainties]
        not_selected_distances = [distance for _, distance, _ in not_selected_questions_distances_uncertainties]
        not_selected_uncertainties = [uncertainty for _, _, uncertainty in not_selected_questions_distances_uncertainties]
        
        not_selected_f1_scores, distances, uncertainties = f1_score(not_selected_distances, not_selected_uncertainties, args)
        print('Length of not selected questions:', len(not_selected_questions_distances_uncertainties))
        print(f'Not selected indexes: {not_selected_questions_idxs}')
        print(f'F1 scores: {not_selected_f1_scores}')
        
        if args.greedy:
            highest_f1_score = max(not_selected_f1_scores)
            selected_idx = not_selected_questions_idxs[np.where(not_selected_f1_scores == highest_f1_score)[0][0]]
            
        else:
            probs = softmax(not_selected_f1_scores)
            print(f'Probs: {probs}')
            print(f'Sum of probs: {sum(probs)}')
            print('------------------')

            customDist = stats.rv_discrete(name='custm', values=(not_selected_questions_idxs, probs))
            selected_idx = customDist.rvs(size=1)[0]

        selected_idxs.append(selected_idx)
        selected_data.append(embeddings[selected_idx])
        demos.append(dataloader[selected_idx])

        index_selected_question = not_selected_questions_idxs.index(selected_idx)
        info_dic = {'iteration_nr': j,
                    'selected_index': int(selected_idx),
                    'current_selected_indexes': [int(el) for el in selected_idxs],
                    'distance' : float(not_selected_distances[index_selected_question]),
                    f'uncertainty_{args.sort_by}' : float(not_selected_uncertainties[index_selected_question]),
                    'f1_score': float(not_selected_f1_scores[index_selected_question])}

        if not args.greedy:
            info_dic['prob'] = float(probs[index_selected_question])
            info_dic['highest_prob'] = float(max(probs))

        if args.normalize_distance_uncertainty:
            info_dic['normalized_distance'] = float(distances[index_selected_question])
            info_dic[f'normalized_uncertainty_{args.sort_by}'] = float(uncertainties[index_selected_question])
        all_info_list.append(info_dic)
        j += 1

        print('Iteration: ', j)
        print('Selected idx: ', selected_idx)
        print(f'Index of selected indices: {index_selected_question}')
        print(f'Selected indices: {selected_idxs}')
        if args.distance_metric == 'cosine':
            print("Number of distances equal to 1: ", len(D2[D2 == 1]))
        else:
            print("Number of distances equal to 0: ", len(D2[D2 == 0]))
        print('*' * 50)

    demos = {"demo": demos}
    with open(args.demos_save_dir + 'demos', 'w', encoding="utf-8") as write_f:
        json.dump(demos, write_f, indent=4, ensure_ascii=False)

    with open(distance_uncertainty_filepath, 'w') as f:
        f.write(json.dumps(all_info_list, indent=2))

if __name__ == "__main__":
    main()
