import json
import argparse
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain.vectorstores import FAISS
from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from constant_vars import *
import datetime
from generate_demo_auto_active_cot_kmeans_plusplus_new import main_auto_active_kmeansplusplus
import time
import os
from utils.load_data import create_dataloader
from utils.prompts_llm import build_prefix, initialize_llmchain, create_prompt_template_gpt35, create_prompt_template_other_models, initialize_llm, from_chatmodelmessages_to_string
from utils.save_results import inference_save_info
from utils.embedding_generation import initialize_embedding_model
from utils.inference_llm import single_question_inference

def parse_arguments():
    parser = argparse.ArgumentParser(description="Auto-Active-CoT-KMeansPlusPlus-Retrieval-Inference")
    parser.add_argument(
        "--dataset", type=str, default="gsm8k",
        choices=["aqua", "gsm8k"], help="dataset used for experiment"
    )

    parser.add_argument(
        "--data_path", type=str, default="../datasets/gsm8k/train.jsonl",
        choices=["../datasets/gsm8k/train.jsonl", "../datasets/AQuA/train.json"], help="dataset used for experiment"
    )

    parser.add_argument(
        "--model_id", type=str, default="gpt-35-turbo-0613", choices=["gpt-35-turbo-0613", "gpt-3.5-turbo-0613", "gpt-4"], help="model used for decoding."
    )

    parser.add_argument(
        "--backup_model_id", type=str, default="gpt-3.5-turbo-0613", choices=["gpt-35-turbo-0613", "gpt-3.5-turbo-0613", "gpt-4"], help="model used for decoding."
    )

    parser.add_argument(
        "--normalize_distance_uncertainty", type=bool, default=True, help="whether to normalize the distance uncertainty before applying F1 score"
    )
    
    parser.add_argument("--random_seed", type=int, default=1, help="random seed")
    parser.add_argument(
        "--num_trails", type=int, default=5, help="number of trails to run for each question"
    )

    parser.add_argument(
        "--method", type=str, default="cot", choices=["zero_shot_cot", "standard", "cot"], help="method"
    )

    parser.add_argument(
        "--max_ra_len", type=int, default=100000000000, help="maximum number of reasoning chains"
    )

    parser.add_argument(
        "--max_token_len", type=int, default=1000000000000, help="maximum number of reasoning chains"
    )

    parser.add_argument(
        "--dir_prompts", type=str, default="uncertainty_estimation_prompts/gsm8k", help="prompts to use for uncertainty estimation"
    )
    
    parser.add_argument(
        "--dataset_size_limit", type=int, default=1000, help="limit the size of training data used to select the demonstrations"
    )
    parser.add_argument(
        "--sort_by", type=str, default='entropy', choices=['disagreement', 'variance', 'entropy'], help="sort the final result by given option"
    )

    parser.add_argument(
        "--embedding_model_id", type=str, default="text-embedding-ada-002-v2", help="the id of the embedding model to use"
    )

    parser.add_argument(
        "--distance_metric", type=str, default='cosine', choices=['cosine', 'euclidean'], help="sort the final result by given option"
    )

    parser.add_argument(
        "--beta", type=int, default=2, help="weight for uncertainty. For example beta=2 means uncertainty is twice as important as the distance"
    )

    parser.add_argument(
        "--top_r", type=int, default=0.1, help="weight for uncertainty. For example beta=2 means uncertainty is twice as important as the distance"
    )
    
    parser.add_argument(
        "--auto_active_kmeansplusplus_temperature", type=float, default=0.7, help="temperature for llm decoding"
    )

    parser.add_argument(
        "--inference_temperature", type=float, default=0.0, help="temperature for llm decoding"
    )

    parser.add_argument(
        "--multipath", type=int, default=1, help="self-consistency path num"
    )

    parser.add_argument(
        "--auto_active_kmeansplusplus_nr_demos", type=int, default=50, help='the number of examples to use for auto-active labeling'
    )

    parser.add_argument(
        "--answers_are_available", type=bool, default=True, help='true if answers are available in the test dataset, false otherwise'
    )

    parser.add_argument(
        "--greedy", type=bool, default=True, help='whether to select examples with the highest f1-score or use random-weighted sampling'
    )

    parser.add_argument(
        "--load_embeddings_file", type=str, default='embeddings/gsm8k/2023_08_29_22_56_01/embeddings.pkl', help='file to load embeddings from'
    )

    parser.add_argument(
        "--load_embeddings_args_file", type=str, default='embeddings/gsm8k/2023_08_29_22_56_01/args.json', help='file to load embeddings from; either None or a path to a file'
    )

    # use the unsorted uncertainty file to select the demonstrations for Auto-Active-KMeansPlusPlus and Auto-Active-KMeansPlusPlus-Retrieval CoT
    parser.add_argument(
        "--load_uncertainty_file", type=str, default='final_uncertainties/2023_08_29_14_44_47/unsorted_all_uncertainty_records', help='file to load uncertainties from'
    )

    parser.add_argument(
        "--load_uncertainty_args_file", type=str, default='final_uncertainties/2023_08_29_14_44_47/args.json', help='nr of demonstrations to select'
    )

    # Retrieval Arguments
    parser.add_argument(
        "--retrieval", type=bool, default=True, help='whether to use retrieval to generate the prompt'
    )

    parser.add_argument(
        "--test_data_path", type=str, default="../datasets/gsm8k/test.jsonl", choices=["../datasets/AQuA/test.json", "../datasets/gsm8k/test.jsonl"],  help="dataset to inference"
    )

    parser.add_argument(
        "--test_dataset_size_limit", type=int, default=0, help='the number of examples to use from the test dataset for inference'
    )

    parser.add_argument(
        "--retrieval_nr_demos", type=int, default=8, help='number of demonstrations'
    )

    parser.add_argument(
        "--output_dir", type=str, default="inference_results", help="output directory"
    )

    # labeled_demos/auto_active_kmeansplusplus/2023_08_12_20_09_07/demos/demos
    
    parser.add_argument(
        "--load_auto_active_kmeansplusplus_demos_file_path", type=str, default='labeled_demos/auto_active_kmeansplusplus/2023_09_07_19_27_28/demos/demos', help="file path of the demonstrations from the auto-active kmeans++ method"
    )

    parser.add_argument(
        "--load_auto_active_kmeansplusplus_metadata_file_path", type=str, default='labeled_demos/auto_active_kmeansplusplus/2023_09_07_19_27_28/metadata/metadata', help="file path of the metadata from the auto-active kmeans++ method"
    )

    parser.add_argument(
        "--load_auto_active_kmeansplusplus_args_file_path", type=str, default='labeled_demos/auto_active_kmeansplusplus/2023_09_07_19_27_28/args.json', help="file path of the args from the auto-active kmeans++ method"
    )

    

    args = parser.parse_args()

    if args.multipath > 1:
        args.inference_temperature = 0.7

    if args.dataset == "gsm8k":
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"

    elif args.dataset == "aqua":
        args.direct_answer_trigger = "\nThe answer is"
    else:
        raise NotImplementedError

    if args.answers_are_available:
        args.demos_save_dir = "labeled_demos"
    else:
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
 
    if args.retrieval:
        current_time = datetime.datetime.now()
        time_string = current_time.strftime("%Y_%m_%d_%H_%M_%S")
        if not os.path.exists(args.demos_save_dir):
            os.makedirs(args.demos_save_dir)
            os.makedirs(args.demos_save_dir + '/' + 'auto_active_kmeansplusplus_retrieval')
            os.makedirs(args.demos_save_dir + '/' +  'auto_active_kmeansplusplus_retrieval' + '/' + time_string)
            os.makedirs(args.demos_save_dir + '/' + 'auto_active_kmeansplusplus_retrieval' + '/' + time_string + '/' + 'auto_active_kmeansplusplus_demos')
            os.makedirs(args.demos_save_dir + '/' + 'auto_active_kmeansplusplus_retrieval' + '/' + time_string + '/' + 'auto_active_kmeansplusplus_metadata')
            os.makedirs(args.demos_save_dir + '/' + 'auto_active_kmeansplusplus_retrieval' + '/' + time_string + '/' + 'test_questions_prompts')
        elif not os.path.exists(args.demos_save_dir + '/' + 'auto_active_kmeansplusplus_retrieval'):
            os.makedirs(args.demos_save_dir + '/' + 'auto_active_kmeansplusplus_retrieval')
            os.makedirs(args.demos_save_dir + '/' +  'auto_active_kmeansplusplus_retrieval' + '/' + time_string)
            os.makedirs(args.demos_save_dir + '/' + 'auto_active_kmeansplusplus_retrieval' + '/' + time_string + '/' + 'auto_active_kmeansplusplus_demos')
            os.makedirs(args.demos_save_dir + '/' + 'auto_active_kmeansplusplus_retrieval' + '/' + time_string + '/' + 'auto_active_kmeansplusplus_metadata')
            os.makedirs(args.demos_save_dir + '/' + 'auto_active_kmeansplusplus_retrieval' + '/' + time_string + '/' + 'test_questions_prompts')
        else:
            os.makedirs(args.demos_save_dir + '/' +  'auto_active_kmeansplusplus_retrieval' + '/' + time_string)
            os.makedirs(args.demos_save_dir + '/' + 'auto_active_kmeansplusplus_retrieval' + '/' + time_string + '/' + 'auto_active_kmeansplusplus_demos')
            os.makedirs(args.demos_save_dir + '/' + 'auto_active_kmeansplusplus_retrieval' + '/' + time_string + '/' + 'auto_active_kmeansplusplus_metadata')
            os.makedirs(args.demos_save_dir + '/' + 'auto_active_kmeansplusplus_retrieval' + '/' + time_string + '/' + 'test_questions_prompts')

        args.args_file = args.demos_save_dir + '/' + 'auto_active_kmeansplusplus_retrieval' + '/' + time_string + '/' + 'args.json'
        args.metadata = args.demos_save_dir + '/' + 'auto_active_kmeansplusplus_retrieval' + '/' + time_string + '/' + 'auto_active_kmeansplusplus_metadata/'
        args.test_questions_prompts_dir = args.demos_save_dir + '/' + 'auto_active_kmeansplusplus_retrieval' + '/' + time_string + '/' + 'test_questions_prompts/'
        args.auto_active_kmeansplusplus_demos_save_dir = args.demos_save_dir + '/' + 'auto_active_kmeansplusplus_retrieval' + '/' + time_string + '/' + 'auto_active_kmeansplusplus_demos/'

        args_dict = {
            "sampling_method": "Auto_Active_KMeansPlusPlus_Retrieval",
            "dataset": args.dataset,
            "data_path": args.data_path,
            "dataset_size_limit": args.dataset_size_limit,
            "random_seed": args.random_seed,
            "nr_demos": args.auto_active_kmeansplusplus_nr_demos, 
            "answers_are_available": args.answers_are_available,
            "demos_save_dir": args.demos_save_dir,
            "greedy": args.greedy,
            "normalize_distance_uncertainty": args.normalize_distance_uncertainty,
            "distance_metric": args.distance_metric,
            "beta": args.beta,
            "retrieval": args.retrieval,
            "test_data_path": args.test_data_path,
            "test_dataset_size_limit": args.test_dataset_size_limit,
            "inference_temperature" : args.inference_temperature,
            "multi_path" : args.multipath,
            'auto_active_limit_nr': args.auto_active_kmeansplusplus_nr_demos,
            "retrieval_nr_demos": args.retrieval_nr_demos,
            "output_dir": args.output_dir,
            "load_auto_active_kmeansplusplus_demos_file_path": args.load_auto_active_kmeansplusplus_demos_file_path,
            "load_auto_active_kmeansplusplus_metadata_file_path": args.load_auto_active_kmeansplusplus_metadata_file_path,
            "load_auto_active_kmeansplusplus_args_file_path": args.load_auto_active_kmeansplusplus_args_file_path,
        }

        start = time.time()

        if args.load_auto_active_kmeansplusplus_demos_file_path and args.load_auto_active_kmeansplusplus_metadata_file_path and args.load_auto_active_kmeansplusplus_args_file_path:
            # use with open to load json files below 
            with open(args.load_auto_active_kmeansplusplus_demos_file_path, 'r', encoding="utf-8") as f:
                demos_json = json.load(f)

            with open(args.load_auto_active_kmeansplusplus_metadata_file_path, 'r', encoding="utf-8") as f:
                auto_active_kmeansplusplus_info_list = json.load(f)

            with open(args.load_auto_active_kmeansplusplus_args_file_path, 'r', encoding="utf-8") as f:
                auto_active_kmeans_args = json.load(f)
            
            args_dict['Auto_Active_KMeansPlusplus_args'] = auto_active_kmeans_args
        else:
            if args.auto_active_kmeansplusplus_nr_demos <= args.retrieval_nr_demos:
                print('The number of examples to use for auto-active labeling should be greater than the number of demonstrations. Proceeding with the auto_active_limit_nr = args.nr_demos - 1.')
                args.auto_active_kmeansplusplus_nr_demos = args.retrieval_nr_demos + 1
            demos_json, auto_active_kmeansplusplus_info_list = main_auto_active_kmeansplusplus(args, args_dict)
        
        with open(args.metadata + 'metadata' , 'w') as f:
            f.write(json.dumps(auto_active_kmeansplusplus_info_list, indent=2))

        with open(args.auto_active_kmeansplusplus_demos_save_dir + 'demos', 'w', encoding="utf-8") as write_f:
            json.dump(demos_json, write_f, indent=4, ensure_ascii=False)


        examples = [{'question' : example['question'],
                    'answer': example['rationale'] + f' The answer is {example["final_answer"]}' + '.\n'
                    } for example in demos_json['demo']]
        
        example_prompt = PromptTemplate(
        input_variables=["question", "answer"],
        template="{question}\n{answer}",
        )

        encoder = initialize_embedding_model(args)

        example_selector = SemanticSimilarityExampleSelector.from_examples(
        examples, 
        encoder, 
        FAISS, 
        k=args.retrieval_nr_demos
        )

        similar_prompt = FewShotPromptTemplate(
            example_selector=example_selector,
            example_prompt=example_prompt,
            suffix="{question}", 
            input_variables=["question"],
        )
  
        args.data_path = args.test_data_path
        args.dataset_size_limit = args.test_dataset_size_limit
        test_dataloader = create_dataloader(args)
        
        correct_nr = 0
        wrong_list = []
        QA_record_list = []
        
        args.temperature = args.inference_temperature
        build_prefix(args)
        args.suffix = "\nQ: " + "{question}" + "\nA: Let's think step by step."
        args.method = 'cot'
        llm = initialize_llm(args, model_id=args.model_id)
        backup_llm = initialize_llm(args, model_id=args.backup_model_id)

        dic = {'gpt-35': create_prompt_template_gpt35, 
               'others' : create_prompt_template_other_models
               } 
        if args.model_id.startswith("gpt-35"):
            model_key = 'gpt-35'
        else:
            model_key = 'others'
        
        is_answer_from_backupmodel_idxs = []
        failed_examples = []
        for test_question_id, test_example in enumerate(test_dataloader):
            try:
                formatted_prompt = similar_prompt.format(question=test_example['question'])
            except Exception as e:
                print(f'ERROR: {e}')
                failed_examples.append({'question_idx': test_question_id, 'question': test_example['question']})
                continue
                
            few_shot_examples = formatted_prompt[:formatted_prompt.rindex('Q:')]
            prompt = args.prefix  + ' Follow the format of the examples below:\n' + few_shot_examples + args.suffix
            prompt_callable = dic[model_key]
            prompt_template = prompt_callable(prompt, args)
            
            # print(f'PROMPT TEMPLATE for question {test_question_id}:')
            # print(from_chatmodelmessages_to_string(prompt_template))            
            # print('*' * 60)
            
            llm_chain = initialize_llmchain(llm, prompt_template)
            backup_llm_chain = initialize_llmchain(backup_llm, prompt_template)  
            correct_nr, wrong_list, QA_record_list, is_answer_from_backupmodel  = single_question_inference(args, test_example, test_question_id, correct_nr, wrong_list, QA_record_list, llm_chain, backup_llm_chain)
        
            if is_answer_from_backupmodel:
                is_answer_from_backupmodel_idxs.append(test_question_id)

            test_q_dic = {
                'test_question_idx' : test_question_id,
                'test_question': test_example['question'],
                'formatted_prompt': few_shot_examples
            }

            with open(f'{args.test_questions_prompts_dir}qes_{test_question_id}' , 'w') as f:
                f.write(json.dumps(test_q_dic, indent=2))

            # except Exception as e:
            #     print(f'Error in question {test_question_id}: {e}')
            #     print(f'Corrent nr : {correct_nr}')
            #     inference_save_info(args, [correct_nr], [wrong_list], [QA_record_list], None, test_question_id + 1)      
            #     break 

        end = time.time()   
        
        args_dict["execution_time"] =  str(end - start) + ' seconds'
        args_dict["used_test_dataset_size"] = len(test_dataloader) - len(failed_examples)

        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
            os.makedirs(args.output_dir + '/' + time_string)
        else:
            os.makedirs(args.output_dir + '/' + time_string)

        args.output_dir = args.output_dir + '/' + time_string + '/'
        with open(args.output_dir + 'args.json', 'w') as f:
            json.dump(args_dict, f, indent=4)

        with open(args.output_dir + 'failed_examples.txt', 'w') as f:
            f.write(json.dumps(failed_examples, indent=4)) 

        with open(args.output_dir + 'answers_backup_model.txt', 'w') as f:
            f.write(json.dumps(is_answer_from_backupmodel_idxs, indent=4))

        inference_save_info(args, [correct_nr], [wrong_list], [QA_record_list], None, len(test_dataloader) - len(failed_examples))      
        
        print('Auto-Active-KMeansPlusPlus-Retrieval Demo Generation and Inference finished.')

    else:
        _, _, = main_auto_active_kmeansplusplus(args, None)



if __name__ == "__main__":
    main()
