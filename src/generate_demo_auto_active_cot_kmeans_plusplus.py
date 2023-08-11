import numpy as np
import json
import os
import pandas as pd
from scipy import stats
from sklearn.metrics import pairwise_distances
import load_env_vars
from constant_vars import *
import datetime
import pickle
import time
from utils.load_data import create_dataloader
from utils.prompts_llm import build_prompt_initialize_llmchain
from utils.uncertainty_estimation import generate_uncertainty_all_questions
from utils.embedding_generation import generate_corpus_embeddings

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

def main_auto_active_kmeansplusplus(args):

    if not args.retrieval:

        current_time = datetime.datetime.now()
        time_string = current_time.strftime("%Y_%m_%d_%H_%M_%S")
        if not os.path.exists(args.demos_save_dir):
            os.makedirs(args.demos_save_dir)
            os.makedirs(args.demos_save_dir + '/' + 'auto_active_kmeansplusplus')
            os.makedirs(args.demos_save_dir + '/' +  'auto_active_kmeansplusplus' + '/' + time_string)
            os.makedirs(args.demos_save_dir + '/' + 'auto_active_kmeansplusplus' + '/' + time_string + '/' + 'demos')
            os.makedirs(args.demos_save_dir + '/' + 'auto_active_kmeansplusplus' + '/' + time_string + '/' + 'metadata')
        elif not os.path.exists(args.demos_save_dir + '/' + 'auto_active_kmeansplusplus'):
            os.makedirs(args.demos_save_dir + '/' + 'auto_active_kmeansplusplus')
            os.makedirs(args.demos_save_dir + '/' +  'auto_active_kmeansplusplus' + '/' + time_string)
            os.makedirs(args.demos_save_dir + '/' + 'auto_active_kmeansplusplus' + '/' + time_string + '/' + 'demos')
            os.makedirs(args.demos_save_dir + '/' + 'auto_active_kmeansplusplus' + '/' + time_string + '/' + 'metadata')
        else:
            os.makedirs(args.demos_save_dir + '/' +  'auto_active_kmeansplusplus' + '/' + time_string)
            os.makedirs(args.demos_save_dir + '/' + 'auto_active_kmeansplusplus' + '/' + time_string + '/' + 'demos')
            os.makedirs(args.demos_save_dir + '/' + 'auto_active_kmeansplusplus' + '/' + time_string + '/' + 'metadata')

        args.args_file = args.demos_save_dir + '/' + 'auto_active_kmeansplusplus' + '/' + time_string + '/' + 'args.json'
        args.metadata = args.demos_save_dir + '/' + 'auto_active_kmeansplusplus' + '/' + time_string + '/' + 'metadata/'
        args.demos_save_dir = args.demos_save_dir + '/' + 'auto_active_kmeansplusplus' + '/' + time_string + '/' + 'demos/'

    dataloader = create_dataloader(args)

    if not args.retrieval:
        start = time.time()
    
    if args.load_uncertainty_file:
        with open(args.load_uncertainty_file, 'r', encoding="utf-8") as read_f:
            all_uncertainty_records = json.load(read_f)['result']
    else:
        args.temperature = args.auto_active_kmeansplusplus_temperature
        build_prompt_initialize_llmchain(args)    
        args.sort = False
        args.dataloader = dataloader
        all_uncertainty_records = generate_uncertainty_all_questions(args)

    questions_idxs = [uncertainty_record['question_id'] for uncertainty_record in all_uncertainty_records]
    uncertainty_list = [uncertainty_record[args.sort_by] for uncertainty_record in all_uncertainty_records]
    
    if args.load_embeddings_file:
        with open(args.load_embeddings_file, 'rb') as read_f:
            corpus_embeddings = pickle.load(read_f)
    else:
        corpus_embeddings = generate_corpus_embeddings(args, dataloader)
        
    assert len(corpus_embeddings) == len(uncertainty_list) == len(questions_idxs)

    uncertainties_series = pd.Series(data=uncertainty_list, index=questions_idxs)
    first_question_idx = list(uncertainties_series.sort_values(ascending=False).head(1).index)[0]
    selected_idxs = [first_question_idx]
    selected_data = [corpus_embeddings[first_question_idx]]
    auto_active_kmeansplusplus_info_list = [{'iteration' : 0,
                     'selected_idx': first_question_idx,
                    }] 
    j = 1

    demos = [dataloader[first_question_idx]]
    print(f'Iteration: {j}')
    print(f'All indices: {questions_idxs}')
    print(f'Selected indices: {selected_idxs}')
    while j < args.auto_active_kmeansplusplus_nr_demos:
        if len(selected_data) == 1:
            D2 = pairwise_distances(corpus_embeddings, selected_data, metric=args.distance_metric, n_jobs=-1).ravel().astype(float)
            if args.distance_metric == 'cosine':
                D2 = 1 - D2
        else:
            newD = pairwise_distances(corpus_embeddings, [selected_data[-1]], metric=args.distance_metric, n_jobs=-1).ravel().astype(float)
            if args.distance_metric == 'cosine':
                newD = 1 - newD
                for i in range(len(corpus_embeddings)):
                    if D2[i] < newD[i]:
                        D2[i] = newD[i]

            elif args.distance_metric == 'euclidean':
                for i in range(len(corpus_embeddings)):
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
            # print(f'Probs: {probs}')
            # print(f'Sum of probs: {sum(probs)}')
            # print('------------------')

            customDist = stats.rv_discrete(name='custm', values=(not_selected_questions_idxs, probs))
            selected_idx = customDist.rvs(size=1)[0]

        selected_idxs.append(selected_idx)
        selected_data.append(corpus_embeddings[selected_idx])
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
        auto_active_kmeansplusplus_info_list.append(info_dic)
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

    demos_json = {"demo": demos}

    if not args.retrieval:
        end = time.time()
        args_dict = {
            "sampling_method": "Auto_Active_KMeansPlusPlus",
            "dataset": args.dataset,
            "data_path": args.data_path,
            "dataset_size_limit": args.dataset_size_limit,
            "dir_prompts": args.dir_prompts,
            "model_id": args.model_id,
            "normalize_distance_uncertainty": args.normalize_distance_uncertainty,
            "random_seed": args.random_seed,
            "num_trails": args.num_trails,
            "method": args.method,
            "sort_by": args.sort_by,
            "distance_metric": args.distance_metric,
            "beta": args.beta,
            "temperature": args.auto_active_kmeansplusplus_temperature,
            "nr_demos": args.auto_active_kmeansplusplus_nr_demos, 
            "answers_are_available": args.answers_are_available,
            "greedy": args.greedy,
            "demos_save_dir": args.demos_save_dir,
            "load_embeddings_file": args.load_embeddings_file,
            "load_uncertainty_file": args.load_uncertainty_file,
            "execution_time": end - start,
        }
        
        with open(args.args_file, 'w') as f:
            json.dump(args_dict, f, indent=4)

        with open(args.demos_save_dir + 'demos', 'w', encoding="utf-8") as write_f:
            json.dump(demos_json, write_f, indent=4, ensure_ascii=False)

        with open(args.metadata + 'metadata' , 'w', encoding='utf-8') as f:
            f.write(json.dumps(auto_active_kmeansplusplus_info_list, indent=2, ensure_ascii=False))

    return demos_json, auto_active_kmeansplusplus_info_list
