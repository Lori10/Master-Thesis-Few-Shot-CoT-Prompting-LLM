import json
import argparse
from utils import *
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain.vectorstores import FAISS
from langchain.prompts import FewShotPromptTemplate, PromptTemplate
import load_env_vars
from constant_vars import *
import datetime
from langchain.embeddings import OpenAIEmbeddings
import openai
from generate_demo_auto_active_cot_kmeans_plusplus import main_auto_active_kmeansplusplus
from inference import single_run_inference

def parse_arguments():
    parser = argparse.ArgumentParser(description="Auto-Active-CoT-KMeansPlusPlus-Retrieval")
    parser.add_argument(
        "--dataset", type=str, default="gsm8k",
        choices=["aqua", "gsm8k"], help="dataset used for experiment"
    )

    parser.add_argument(
        "--data_path", type=str, default="../datasets/gsm8k/train.jsonl",
        choices=["../datasets/gsm8k/train.jsonl", "../datasets/AQuA/train.json"], help="dataset used for experiment"
    )

    parser.add_argument(
        "--model_id", type=str, default="text-davinci-003", choices=["text-davinci-003", "tiiuae/falcon-7b-instruct"], help="model used for decoding."
    )

    parser.add_argument(
        "--normalize_distance_uncertainty", type=bool, default=False, help="whether to normalize the distance uncertainty before applying F1 score"
    )
    
    parser.add_argument("--random_seed", type=int, default=42, help="random seed")
    parser.add_argument(
        "--num_trails", type=int, default=3, help="number of trails to run for each question"
    )

    parser.add_argument(
        "--method", type=str, default="few_shot_cot", choices=["zero_shot_cot", "few_shot_cot"], help="method"
    )

    parser.add_argument(
        "--dir_prompts", type=str, default="uncertainty_estimation_prompts/gsm8k", help="prompts to use for uncertainty estimation"
    )
    
    parser.add_argument(
        "--dataset_size_limit", type=int, default=20, help="limit the size of training data used to select the demonstrations"
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
        "--auto_active_kmeansplusplus_nr_demos", type=int, default=3, help='the number of examples to use for auto-active labeling'
    )

    parser.add_argument(
        "--retrieval_nr_demos", type=int, default=3, help='number of demonstrations'
    )

    parser.add_argument(
        "--answers_are_available", type=bool, default=True, help='true if answers are available in the test dataset, false otherwise'
    )

    parser.add_argument(
        "--greedy", type=bool, default=True, help='whether to select examples with the highest f1-score or use random-weighted sampling'
    )

    parser.add_argument(
        "--test_data_path", type=str, default="../datasets/gsm8k/test.jsonl", choices=["../datasets/AQuA/test.json", "../datasets/gsm8k/test.jsonl"],  help="dataset to inference"
    )

    # test_question_gsm8k = 'Dave bought a large pack of french fries and ate fourteen before a hungry seagull stole the pack out of his hand. When the seagull landed, he gobbled down half the amount of french fries that Dave ate. Then three pigeons bullied him away from the food, and each pigeon ate three fries. Later, a raccoon stole two thirds of the remaining fries. Ants carried off a final french fry, leaving five behind. How many french fries were in the pack when Dave bought it?
    # parser.add_argument(
    #     "--test_question", type=str, default='Dave bought a large pack of french fries and ate fourteen before a hungry seagull stole the pack out of his hand. When the seagull landed, he gobbled down half the amount of french fries that Dave ate. Then three pigeons bullied him away from the food, and each pigeon ate three fries. Later, a raccoon stole two thirds of the remaining fries. Ants carried off a final french fry, leaving five behind. How many french fries were in the pack when Dave bought it?', help='test question for few-shot cot'
    # )

    parser.add_argument(
        "--test_dataset_size_limit", type=int, default=10, help='the number of examples to use from the test dataset for inference'
    )

    parser.add_argument(
        "--output_dir", type=str, default="inference_results", help="output directory"
    )

    parser.add_argument(
        "--retrieval", type=bool, default=False, help='whether to use retrieval to generate the prompt'
    )

    parser.add_argument(
        "--load_demos_auto_active_kmeansplusplus", type=bool, default=True, help="whether to load the demonstrations from the auto-active kmeans++ method or compute the demonstrations from scratch"
    )

    parser.add_argument(
        "--load_demos_auto_active_kmeansplusplus_file_path", type=str, default='labeled_demos/auto_active_kmeansplusplus/2023_08_08_13_01_17/demos/demos', help="file path of the demonstrations from the auto-active kmeans++ method"
    )

    parser.add_argument(
        "--load_demos_auto_active_kmeansplusplus_metadata_file_path", type=str, default='labeled_demos/auto_active_kmeansplusplus/2023_08_08_13_01_17/metadata/metadata', help="file path of the demonstrations from the auto-active kmeans++ method"
    )

    # gsm8k embeddings: 'embeddings/gsm8k/2023_08_10_11_45_01/embeddings.pkl'
    parser.add_argument(
        "--load_embeddings_file", type=str, default=None, help='file to load embeddings from'
    )

    # use the unsorted uncertainty file to select the demonstrations for Auto-Active-KMeansPlusPlus and Auto-Active-KMeansPlusPlus-Retrieval CoT
    parser.add_argument(
        "--load_uncertainty_file", type=str, default='all_uncertainties/gsm8k/unsorted_all_uncertainty_records', help='nr of demonstrations to select'
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
        args.demos_save_dir = "unlabeled_demos"

    # "Therefore, the answer ..." -> "The answer ..."
    trigger = args.direct_answer_trigger.replace("\nTherefore, ", "")
    args.direct_answer_trigger_for_zeroshot = trigger[0].upper() + trigger[1:]
    args.direct_answer_trigger_for_zeroshot_cot = args.direct_answer_trigger
    args.direct_answer_trigger_for_fewshot = "The answer is"
    args.cot_trigger = "Let's think step by step."
    return args

def main_auto_active_kmeansplusplus_retrieval():
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

        args.json_file = args.demos_save_dir + '/' + 'auto_active_kmeansplusplus_retrieval' + '/' + time_string + '/' + 'args.json'
        args.metadata = args.demos_save_dir + '/' + 'auto_active_kmeansplusplus_retrieval' + '/' + time_string + '/' + 'auto_active_kmeansplusplus_metadata/'
        args.test_questions_prompts_dir = args.demos_save_dir + '/' + 'auto_active_kmeansplusplus_retrieval' + '/' + time_string + '/' + 'test_questions_prompts/'
        args.auto_active_kmeansplusplus_demos_save_dir = args.demos_save_dir + '/' + 'auto_active_kmeansplusplus_retrieval' + '/' + time_string + '/' + 'auto_active_kmeansplusplus_demos/'

        args_dict = {
            "sampling_method": "Auto_Active_KMeansPlusPlus_Retrieval",
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
            "temperature_auto_active_kmeansplusplus": args.auto_active_kmeansplusplus_temperature,
            "temperature_inference": args.inference_temperature,
            'auto_active_limit_nr': args.auto_active_kmeansplusplus_nr_demos,
            "retrieval_nr_demos": args.retrieval_nr_demos,
            "answers_are_available": args.answers_are_available,
            "greedy": args.greedy,
            "test_data_path": args.test_data_path,
            "test_dataset_size_limit": args.test_dataset_size_limit, 
            "output_dir": args.output_dir,
            "load_demos_auto_active_kmeansplusplus": args.load_demos_auto_active_kmeansplusplus,
            "load_demos_auto_active_kmeansplusplus_file_path": args.load_demos_auto_active_kmeansplusplus_file_path,
            "load_demos_auto_active_kmeansplusplus_metadata_file_path": args.load_demos_auto_active_kmeansplusplus_metadata_file_path,
            "load_embeddings_file": args.load_embeddings_file
        }

        with open(args.json_file, 'w') as f:
            json.dump(args_dict, f, indent=4)

        start = time.time()

        if args.load_demos_auto_active_kmeansplusplus:
            # use with open to load json files below 
            with open(args.load_demos_auto_active_kmeansplusplus_file_path, 'r', encoding="utf-8") as f:
                demos_json = json.load(f)
            with open(args.load_demos_auto_active_kmeansplusplus_metadata_file_path, 'r', encoding="utf-8") as f:
                auto_active_kmeansplusplus_info_list = json.load(f)

            openai.api_key = os.getenv("OPENAI_API_KEY")
            headers = {
                "x-api-key": openai.api_key,
            }
            encoder = OpenAIEmbeddings(
                deployment="text-embedding-ada-002-v2", headers=headers, chunk_size=1
            )
        else:
            if args.auto_active_kmeansplusplus_nr_demos <= args.retrieval_nr_demos:
                print('The number of examples to use for auto-active labeling should be greater than the number of demonstrations. Proceeding with the auto_active_limit_nr = args.nr_demos - 1.')
                args.auto_active_kmeansplusplus_nr_demos = args.retrieval_nr_demos + 1
            demos_json, encoder, auto_active_kmeansplusplus_info_list = main_auto_active_kmeansplusplus(args)
        
        with open(args.metadata + 'metadata' , 'w') as f:
            f.write(json.dumps(auto_active_kmeansplusplus_info_list, indent=2))

        with open(args.auto_active_kmeansplusplus_demos_save_dir + 'demos', 'w', encoding="utf-8") as write_f:
            json.dump(demos_json, write_f, indent=4, ensure_ascii=False)


        if args.dataset == "gsm8k":
            prefix = prefix_gsm8k
        elif args.dataset == "aqua":
            prefix = prefix_aqua
        else:
            raise NotImplementedError("dataset not implemented")

        examples = [{'question' : example['question'],
                    'answer': example['rationale'] + f' The answer is {example["final_answer"]}' + '.\n'
                    } for example in demos_json['demo']]
        
        example_prompt = PromptTemplate(
        input_variables=["question", "answer"],
        template="{question}\n{answer}",
        )

        example_selector = SemanticSimilarityExampleSelector.from_examples(
        examples, 
        encoder, 
        FAISS, 
        k=args.retrieval_nr_demos
        )

        similar_prompt = FewShotPromptTemplate(
            example_selector=example_selector,
            example_prompt=example_prompt,
            suffix="Split:" + "{question}", 
            input_variables=["question"],
        )
        
        args.data_path = args.test_data_path
        test_dataloader = create_dataloader(args)
        if args.test_dataset_size_limit <= 0:
            args.test_dataset_size_limit = len(test_dataloader)
        else:
            test_dataloader = test_dataloader[:args.test_dataset_size_limit]
        
        correct_nr = 0
        wrong_list = []
        QA_record_list = []

        args.temperature = args.inference_temperature
        for test_question_id, test_example in enumerate(test_dataloader):
            few_shot_examples = similar_prompt.format(question=test_example['question']).split('Split:')[0]
            #print('Test Question ID: ' + str(test_question_id) + '\n')
            #print(f'Prompt:\n{few_shot_examples}')
            print('*' * 60)

            full_prompt = prefix  + ' To generate the answer follow the format of the examples below:\n' + few_shot_examples + "\nQ: " + "{question}" + "\nA: Let's think step by step."
            args.llm_chain = initialize_llmchain(full_prompt, args)
            correct_nr, wrong_list, QA_record_list = single_question_inference(args, test_example, test_question_id, correct_nr, wrong_list, QA_record_list)
            
            test_q_dic = {
                'test_question_idx' : test_question_id,
                'test_question': test_example['question'],
                'formatted_prompt': few_shot_examples
            }

            with open(f'{args.test_questions_prompts_dir}qes_{test_question_id}' , 'w') as f:
                f.write(json.dumps(test_q_dic, indent=2))

        end = time.time()
        print('Total Execution Time: ', end - start, " seconds")

        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
            os.makedirs(args.output_dir + '/' + time_string)
        else:
            os.makedirs(args.output_dir + '/' + time_string)

        args.output_dir = args.output_dir + '/' + time_string + '/'
        inference_save_info(args, [correct_nr], [wrong_list], [QA_record_list], None, len(test_dataloader))        
    else:
        _, _, _ = main_auto_active_kmeansplusplus(args)

    print('Inference finished.')


if __name__ == "__main__":
    main_auto_active_kmeansplusplus_retrieval()
