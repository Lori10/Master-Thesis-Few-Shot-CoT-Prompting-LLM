import argparse
from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain.vectorstores import FAISS
from utils.load_data import create_dataloader 
import time
import json
from constant_vars import *
from utils.prompts_llm import initialize_llm, build_prefix, create_prompt_template_gpt35, initialize_llmchain
from utils.save_results import inference_save_info
from utils.embedding_generation import initialize_embedding_model
from utils.inference_llm import single_question_inference
import datetime
import os 
import sys 

def parse_arguments():
    parser = argparse.ArgumentParser(description="Retrieval_CoT")
    parser.add_argument(
        "--dataset", type=str, default="aqua", choices=["gsm8k", "aqua"], help="dataset to inference"
    )

    parser.add_argument(
        "--data_path", type=str, default="../datasets/AQuA/train.json",
        choices=["../datasets/gsm8k/train.jsonl", "../datasets/AQuA/train.json"], help="dataset used for experiment"
    )

    parser.add_argument(
        "--model_id", type=str, default="gpt-35-turbo-0613", choices=["gpt-35-turbo-0613", "gpt-3.5-turbo-0613", "gpt-4"], help="model used for decoding."
    )

    parser.add_argument(
        "--backup_model_id", type=str, default="gpt-3.5-turbo-0613", choices=["gpt-35-turbo-0613", "gpt-3.5-turbo-0613", "gpt-4"], help="model used for decoding."
    )
    
    parser.add_argument(
        "--method", type=str, default="cot", choices=["standard", "zero_shot_cot", "cot"], help="method"
    )
    
    parser.add_argument(
        "--dataset_size_limit", type=int, default=1000, help="whether to limit dataset size. if 0, the dataset size is unlimited and we use all the samples in the dataset for creating the demonstrations."
    )

    parser.add_argument("--random_seed", type=int, default=1, help="random seed")
    
    parser.add_argument(
        "--temperature", type=float, default=0.0, help="temperature for llm decoding"
    )

    parser.add_argument(
        "--nr_demos", type=int, default=8, help='nr of demonstrations to select'
    )

    parser.add_argument(
        "--embedding_model_id", type=str, default="text-embedding-ada-002-v2", help="the id of the embedding model to use"
    )

    parser.add_argument(
        "--test_data_path", type=str, default="../datasets/gsm8k/test.jsonl", choices=["../datasets/AQuA/test.json", "../datasets/gsm8k/test.jsonl"],  help="dataset to inference"
    )

    parser.add_argument(
        "--test_dataset_size_limit", type=int, default=0, help='the number of examples to use from the test dataset for inference'
    )

    parser.add_argument(
        "--answers_are_available", type=bool, default=True, help='true if answers are available in the test dataset, false otherwise'
    )

    parser.add_argument(
        "--output_dir", type=str, default="inference_results", help="output directory"
    )

    parser.add_argument(
        "--multipath", type=int, default=1, help="self-consistency path num"
    )
    
    args = parser.parse_args()
    
    # Fill in the dataset path
    if args.dataset == "gsm8k":
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif args.dataset == "aqua":
        args.direct_answer_trigger = "\nThe answer is"
    else:
        raise ValueError("dataset is not properly defined ...")

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
    current_time = datetime.datetime.now()
    time_string = current_time.strftime("%Y_%m_%d_%H_%M_%S")

    train_dataloader = create_dataloader(args)

    start = time.time()
    
    train_examples = [{'question' : example['question'],
                        'answer': example['rationale'] + f' The answer is {example["final_answer"]}' + '.\n'
                        } for example in train_dataloader]
    
    example_prompt = PromptTemplate(
    input_variables=["question", "answer"],
    template="{question}\n{answer}",
    )

    encoder = initialize_embedding_model(args)
    
    example_selector = SemanticSimilarityExampleSelector.from_examples(
                    train_examples, 
                    encoder, 
                    FAISS, 
                    k=args.nr_demos
    )


    similar_prompt = FewShotPromptTemplate(
        example_selector=example_selector,
        example_prompt=example_prompt,
        suffix="{question}", 
        input_variables=["question"]
    )

    args.data_path = args.test_data_path
    args.dataset_size_limit = args.test_dataset_size_limit
    test_dataloader = create_dataloader(args)

    build_prefix(args)
    args.suffix = "\nQ: " + "{question}" + "\nA: Let's think step by step."

    llm = initialize_llm(args, model_id=args.model_id)
    backup_llm = initialize_llm(args, model_id=args.backup_model_id)

    correct_nr = 0
    wrong_list = []
    QA_record_list = []
    is_answer_from_backupmodel_idxs = []
    failed_examples = []
    for test_question_id, test_example in enumerate(test_dataloader):
        print(f'QUESTION IDX: {test_question_id}')
        print(f'QUESTION: ')
        print(test_example['question'])
        try:
            formatted_prompt = similar_prompt.format(question=test_example['question'])
        except Exception as e:
            print(f'ERROR: {e}')
            failed_examples.append({'question_idx': test_question_id, 'question': test_example['question']})
            continue

        few_shot_examples = formatted_prompt[:formatted_prompt.rindex('Q:')]
    
        prompt = args.prefix  + ' Follow the format of the examples below:\n' + few_shot_examples + args.suffix
        prompt_template = create_prompt_template_gpt35(prompt, args)

        llm_chain = initialize_llmchain(llm, prompt_template)
        backup_llm_chain = initialize_llmchain(backup_llm, prompt_template) 
        correct_nr, wrong_list, QA_record_list, is_answer_from_backupmodel = single_question_inference(args, test_example, test_question_id, correct_nr, wrong_list, QA_record_list, llm_chain, backup_llm_chain)
        if is_answer_from_backupmodel:
            is_answer_from_backupmodel_idxs.append(test_question_id)

        # test_q_dic = {
        #         'test_question_idx' : test_question_id,
        #         'test_question': test_example['question'],
        #         'formatted_prompt': few_shot_examples
        #     }

        # with open(f'{args.test_questions_prompts_dir}qes_{test_question_id}' , 'w') as f:
        #     f.write(json.dumps(test_q_dic, indent=2))

        # except Exception as e:
        #     print(f'Error in question {test_question_id}: {e}')

        #     print('FORMATTED   PROMPT')
        #     print(formatted_prompt)
        #     print('----------------')
        #     print(f'Corrent nr : {correct_nr}')
        #     inference_save_info(args, [correct_nr], [wrong_list], [QA_record_list], None, test_question_id + 1)      
        #     break 

    end = time.time()

    args_dict = {
        'sampling_method': "Retrieval-CoT",
        'dataset': args.dataset,
        'dataset_size_limit': args.dataset_size_limit,
        'model_id': args.model_id,
        'backup_model_id': args.backup_model_id,
        'method': args.method,
        'random_seed': args.random_seed,
        'temperature': args.temperature,
        'nr_demos': args.nr_demos,
        'embedding_model_id': args.embedding_model_id,
        'test_data_path': args.test_data_path,
        'test_dataset_size_limit': args.test_dataset_size_limit,
        'used_test_dataset_size_limit': len(test_dataloader) - len(failed_examples),
        "execution_time": str(end - start) + ' seconds',
    }

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        os.makedirs(args.output_dir + '/' + time_string)
    else:
        os.makedirs(args.output_dir + '/' + time_string)

    args.output_dir = args.output_dir + '/' + time_string + '/'

    with open(args.output_dir + 'args.json', 'w') as f:
        json.dump(args_dict, f, indent=4)

    with open(args.output_dir + 'answers_backup_model.txt', 'w') as f:
        f.write(json.dumps(is_answer_from_backupmodel_idxs, indent=4))

    with open(args.output_dir + 'failed_examples.txt', 'w') as f:
        f.write(json.dumps(failed_examples, indent=4))    

    inference_save_info(args, [correct_nr], [wrong_list], [QA_record_list], None, len(test_dataloader) - len(failed_examples))      

    print('Retrieval-CoT Demo Generation and Inference finished.')


if __name__ == "__main__":
    main()