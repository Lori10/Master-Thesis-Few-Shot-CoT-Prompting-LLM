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
from utils.uncertainty_estimation import generate_uncertainty_all_questions
from utils.embedding_generation import generate_corpus_embeddings
from utils.scaler_and_metrics import f1_score, softmax
from utils.prompts_llm import create_prompts_inference, initialize_llmchain

def main_auto_active_kmeansplusplus(args, args_dict):

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
        args_dict = {
            "sampling_method": "Auto_Active_KMeansPlusPlus",
            "dataset": args.dataset,
            "data_path": args.data_path,
            "dataset_size_limit": args.dataset_size_limit,
            "random_seed": args.random_seed,
            "nr_demos": args.auto_active_kmeansplusplus_nr_demos, 
            "answers_are_available": args.answers_are_available,
            "demos_save_dir": args.demos_save_dir,
            "load_embeddings_file": args.load_embeddings_file,
            "load_embeddings_args_file": args.load_embeddings_args_file,
            "load_uncertainty_file": args.load_uncertainty_file,
            "load_uncertainty_args_file": args.load_uncertainty_args_file,
            "greedy": args.greedy,
            "normalize_distance_uncertainty": args.normalize_distance_uncertainty,
            "distance_metric": args.distance_metric,
            "beta": args.beta
        }

        start = time.time()
    
    if args.load_uncertainty_file and args.load_uncertainty_args_file:
        with open(args.load_uncertainty_file, 'r', encoding="utf-8") as read_f:
            all_uncertainty_records = json.load(read_f)['result']

        with open(args.load_uncertainty_args_file, 'r', encoding="utf-8") as f:
            uncertainty_args = json.load(f)
        
        args_dict['generate_uncertainty_args'] = uncertainty_args
    else:
        args_dict["method"] = args.method
        args_dict["model_id"] = args.model_id
        args_dict["num_trails"] = args.num_trails
        args_dict["sort_by"] = args.sort_by
        args_dict["temperature"] = args.auto_active_kmeansplusplus_temperature
        args_dict["dir_prompts"] = args.dir_prompts

        args.temperature = args.auto_active_kmeansplusplus_temperature
        
        prompts_list = create_prompts_inference(args)
        assert len(prompts_list) == 1
        initialize_llmchain(args, prompts_list[0], llm_init=False)   
        all_uncertainty_records = generate_uncertainty_all_questions(args, dataloader, False)
    
    questions_idxs = [uncertainty_record['question_idx'] for uncertainty_record in all_uncertainty_records]
    uncertainty_list = [uncertainty_record[args.sort_by] for uncertainty_record in all_uncertainty_records]
    
    if args.load_embeddings_file and args.load_embeddings_args_file:
        with open(args.load_embeddings_file, 'rb') as read_f:
            corpus_embeddings = pickle.load(read_f)

        with open(args.load_embeddings_args_file, 'r', encoding="utf-8") as f:
            embeddings_args = json.load(f)

        args_dict['generate_embeddings_args'] = embeddings_args
    else:
        args_dict['embedding_model_id'] = args.embedding_model_id
        corpus_embeddings = generate_corpus_embeddings(args, dataloader)
        
    assert len(corpus_embeddings) == len(uncertainty_list) == len(questions_idxs)

    uncertainties_series = pd.Series(data=uncertainty_list, index=questions_idxs)
    first_question_idx = list(uncertainties_series.sort_values(ascending=False).head(1).index)[0]
    selected_idxs = [first_question_idx]
    selected_data = [corpus_embeddings[first_question_idx]]
    auto_active_kmeansplusplus_info_list = [{'iteration' : 0,
                     'selected_idx': first_question_idx,
                     'uncertainty' : uncertainties_series[first_question_idx],
                     f'highest_uncertainty_{args.sort_by}' : max(uncertainty_list),
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
        args_dict["execution_time"] = str(end - start) + " seconds"
        
        with open(args.args_file, 'w') as f:
            json.dump(args_dict, f, indent=4)

        with open(args.demos_save_dir + 'demos', 'w', encoding="utf-8") as write_f:
            json.dump(demos_json, write_f, indent=4, ensure_ascii=False)

        with open(args.metadata + 'metadata' , 'w', encoding='utf-8') as f:
            f.write(json.dumps(auto_active_kmeansplusplus_info_list, indent=2, ensure_ascii=False))

        print('Auto Active KMeans++ Demo Generation Completed!')

    return demos_json, auto_active_kmeansplusplus_info_list
