import numpy as np
import json
import os
import pandas as pd
from scipy import stats
from sklearn.metrics import pairwise_distances
from constant_vars import *
import datetime
import pickle
import time
from utils.load_data import create_dataloader
from utils.uncertainty_estimation import generate_uncertainty_all_questions
from utils.embedding_generation import generate_corpus_embeddings
from utils.scaler_and_metrics import f1_score, softmax
from utils.prompts_llm import create_prompts_inference, initialize_llmchain, initialize_llm
from utils.filter_simple_examples import filter_examples_with_labels, filter_examples_no_labels
import sys 

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

    if not args.retrieval:
        args_dict = {
            "sampling_method": "Auto_Active_KMeansPlusPlus",
            "dataset": args.dataset,
            "data_path": args.data_path,
            "dataset_size_limit": args.dataset_size_limit,
            "random_seed": args.random_seed,
            "original_nr_demos": args.auto_active_kmeansplusplus_nr_demos, 
            "answers_are_available": args.answers_are_available,
            "demos_save_dir": args.demos_save_dir,
            "load_embeddings_file": args.load_embeddings_file,
            "load_embeddings_args_file": args.load_embeddings_args_file,
            "load_uncertainty_file": args.load_uncertainty_file,
            "load_uncertainty_args_file": args.load_uncertainty_args_file,
            "greedy": args.greedy,
            "normalize_distance_uncertainty": args.normalize_distance_uncertainty,
            "distance_metric": args.distance_metric,
            "beta": args.beta,
            'max_ra_len': args.max_ra_len,
            'top_r': args.top_r
        }

        start = time.time()
    
    if args.load_uncertainty_file and args.load_uncertainty_args_file:
        with open(args.load_uncertainty_file, 'r', encoding="utf-8") as read_f:
            #all_uncertainty_records = json.load(read_f)['result']
            all_uncertainty_records = json.load(read_f)['result'][:args.dataset_size_limit]

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
        
        dataloader = create_dataloader(args)
        prompts_list = create_prompts_inference(args)
        assert len(prompts_list) == 1
        azure_llm = initialize_llm(args, is_azureopenai=True)
        azure_llm_chain = initialize_llmchain(azure_llm, prompts_list[0])
        openai_llm = initialize_llm(args, is_azureopenai=False)
        openai_llm_chain = initialize_llmchain(openai_llm, prompts_list[0])  
        all_uncertainty_records = generate_uncertainty_all_questions(args, dataloader, False, azure_llm_chain, openai_llm_chain)
    
    
    if args.load_embeddings_file and args.load_embeddings_args_file:
        with open(args.load_embeddings_file, 'rb') as read_f:
            # corpus_embeddings = pickle.load(read_f)
            corpus_embeddings = pickle.load(read_f)[:args.dataset_size_limit]

        with open(args.load_embeddings_args_file, 'r', encoding="utf-8") as f:
            embeddings_args = json.load(f)

        args_dict['generate_embeddings_args'] = embeddings_args
    else:
        args_dict['embedding_model_id'] = args.embedding_model_id
        corpus_embeddings = generate_corpus_embeddings(args, dataloader)
    

    print('Total nr of examples: ', len(all_uncertainty_records))
    filtered_uncertainty_records = filter_examples_with_labels(all_uncertainty_records)
    filtered_idxs = [x['question_idx'] for x in filtered_uncertainty_records]
    print(f'Nr of filtered examples: {len(filtered_uncertainty_records)}. Nr of demos: {args.auto_active_kmeansplusplus_nr_demos}')
    if args.auto_active_kmeansplusplus_nr_demos >= len(all_uncertainty_records):
        print('Warning: the number of demos must be lower than the number of filtered examples with labels. Setting the number of demos to the half of number of filtered examples.')
        args.auto_active_kmeansplusplus_nr_demos = round(len(filtered_uncertainty_records) / 2)
        print('Procceding with the number of demos: ', args.auto_active_kmeansplusplus_nr_demos)

    corpus_embeddings = corpus_embeddings[filtered_idxs]
    
    max_entropy_example = max(filtered_uncertainty_records, key=lambda x: x[args.sort_by])
    question_idx_with_max_entropy = filtered_uncertainty_records.index(max_entropy_example)

    selected_idxs = [question_idx_with_max_entropy]
    selected_data = [corpus_embeddings[question_idx_with_max_entropy]]
    auto_active_kmeansplusplus_info_list = [{'iteration' : 0,
                     'selected_idx': question_idx_with_max_entropy,
                     'original_selected_idx': max_entropy_example['question_idx'],
                     'uncertainty' : max_entropy_example[args.sort_by],
                     f'highest_uncertainty_{args.sort_by}' : max([x[args.sort_by] for x in filtered_uncertainty_records])
                    }] 
    j = 1

    demos = [filtered_uncertainty_records[question_idx_with_max_entropy]]
    print(f'Iteration: {j}')
    print(f'All indices: {[idx for idx in range(len(filtered_uncertainty_records))]}')
    print(f'Original All indices: {[example["question_idx"] for example in filtered_uncertainty_records]}')
    print(f'Uncertainty Scores: {[round(example[args.sort_by], 2) for example in filtered_uncertainty_records]}')
    print(f'Selected indices: {selected_idxs}')
    print(f'Original Selected indices: {[filtered_uncertainty_records[i]["question_idx"] for i in selected_idxs]}')
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
            not_selected_examples = [(example, np.exp(-distance), index) for index, (example, distance) in enumerate(zip(filtered_uncertainty_records, D2)) if distance != 1]
        elif args.distance_metric == 'euclidean':
            D2[D2 < 0.0001] = 0
            not_selected_examples = [(example, distance, index) for index, (example, distance) in enumerate(zip(filtered_uncertainty_records, D2)) if distance != 0]

        # not_selected_question_idxs = []
        # not_selected_distances = []
        # not_selected_uncertainties = []
        # for example, distance in not_selected_examples:
        #     not_selected_question_idxs.append(example['question_idx'])
        #     not_selected_distances.append(distance)
        #     not_selected_uncertainties.append(example[args.sort_by])

        not_selected_question_idxs = []
        not_selected_distances = []
        not_selected_uncertainties = []
        for example, distance, idx in not_selected_examples:
            not_selected_question_idxs.append(idx)
            not_selected_distances.append(distance)
            not_selected_uncertainties.append(example[args.sort_by])
            
        not_selected_f1_scores, distances, uncertainties = f1_score(not_selected_distances, not_selected_uncertainties, args)
        
        #not_selected_f1_scores, distances, uncertainties = f1_score(not_selected_distances, not_selected_uncertainties, args)
        print('Length of not selected questions:', len(not_selected_examples))
        print(f'Not selected indexes: {not_selected_question_idxs}')
        print(f'F1 scores: {[round(score, 2) for score in not_selected_f1_scores]}')
        
        if args.greedy:
            highest_f1_score = max(not_selected_f1_scores)
            selected_idx = not_selected_question_idxs[np.where(not_selected_f1_scores == highest_f1_score)[0][0]]
        else:
            sorted_idxs = np.argsort(not_selected_f1_scores)[::-1]
            top_k = round(args.top_r * len(not_selected_f1_scores))
            selected_top_k_idxs = sorted_idxs[:top_k]
            filtered_question_idxs = np.array(not_selected_question_idxs)[selected_top_k_idxs]
            filtered_f1_scores = np.array(not_selected_f1_scores)[selected_top_k_idxs]

            probs = softmax(filtered_f1_scores)
            # print(f'Probs: {probs}')
            # print(f'Sum of probs: {sum(probs)}')
            # print('------------------')

            customDist = stats.rv_discrete(name='custm', values=(filtered_question_idxs, probs))
            selected_idx = customDist.rvs(size=1)[0]
            
        selected_idxs.append(selected_idx)
        selected_data.append(corpus_embeddings[selected_idx])
        demos.append(filtered_uncertainty_records[selected_idx])

        # the code below is needed to store metadata about the selected example
        if args.greedy:
            index_selected_question = not_selected_question_idxs.index(selected_idx)
        else:
            index_selected_question = filtered_question_idxs.tolist().index(selected_idx)

        info_dic = {'iteration_nr': j,
                    'selected_index': int(selected_idx),
                    'original_selected_index': int(filtered_uncertainty_records[selected_idx]['question_idx']),
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
        print(f'Original selected indices: {[filtered_uncertainty_records[i]["question_idx"] for i in selected_idxs]}')
        if args.distance_metric == 'cosine':
            print("Number of distances equal to 1: ", len(D2[D2 == 1]))
        else:
            print("Number of distances equal to 0: ", len(D2[D2 == 0]))
        print('*' * 50)

    demos_json = {"demo": demos}

    if not args.retrieval:
        end = time.time()

        args_dict['used_nr_demos'] = args.auto_active_kmeansplusplus_nr_demos
        args_dict['nr_filtered_examples'] = len(filtered_uncertainty_records)
        args_dict["execution_time"] = str(end - start) + " seconds"
        
        with open(args.args_file, 'w') as f:
            json.dump(args_dict, f, indent=4)

        with open(args.demos_save_dir + 'demos', 'w', encoding="utf-8") as write_f:
            json.dump(demos_json, write_f, indent=4, ensure_ascii=False)

        with open(args.metadata + 'metadata' , 'w', encoding='utf-8') as f:
            f.write(json.dumps(auto_active_kmeansplusplus_info_list, indent=2, ensure_ascii=False))

        print('Auto Active KMeans++ Demo Generation Completed!')

    return demos_json, auto_active_kmeansplusplus_info_list
