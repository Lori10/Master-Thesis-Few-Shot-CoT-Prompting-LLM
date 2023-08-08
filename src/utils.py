# This file contains necessary helper functions
# e.g. GPT request, create_dataloader
import openai
import random
import sys
import numpy as np
from typing import Tuple
import torch
import json
import re
from collections import Counter
import time
import os
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.callbacks import get_openai_callback 
from langchain.llms import HuggingFacePipeline
from langchain.llms import AzureOpenAI
import load_env_vars
from scipy.stats import entropy
import numpy as np

# define for no solution if GPT cannot generate a valid solution
# here define a magic number for the convenience of variance calculation
NO_SOLUTION = '-10086'

def initialize_llmchain(prompt_template: str, args) -> LLMChain:
    """
        Run a LLMChain for given a prompt template and question. Return the completion and
        total nr of processed tokens during the run.
        
        Args: 
            prompt_template (str): the prompt template to use
            args: the arguments passed in from the command line
        Returns:
            llm_chain (LLMChain): the LLMChain object
    """
    prompt = PromptTemplate(input_variables=["question"], template=prompt_template)

    if args.model_id.startswith("gpt-35") or args.model_id.startswith('text-davinci') or args.model_id.startswith("gpt-4"):        
        openai.api_key = os.getenv("OPENAI_API_KEY")
        headers = {
            "x-api-key": openai.api_key,
        }

        llm = AzureOpenAI(
        deployment_name=args.model_id,
        model_name=args.model_id,
        temperature=args.temperature,
        headers=headers,
        max_tokens=1024,
        )
    else:
        llm = HuggingFacePipeline.from_model_id(
        model_id=args.model_id,
        model_kwargs={"temperature": args.temperature,
                     "trust_remote_code": True,
                     "max_seq_len": 4096}, # max_length
        )
    
    llm_chain = LLMChain(prompt=prompt, llm=llm, verbose=False)
    return llm_chain

def set_random_seed(seed: int):
    """
        Fix the seed for random, numpy and pytorch
        Args:
            seed (int): the seed to fix
    """
    # random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
def load_data(args: object) -> Tuple[list, list, list]:
    """
        Load the data from the given path
        Args:
            args (object): the arguments passed in from the command line
        Returns:
            questions (list): the list of questions
            rationales (list): the list of rationales
            final_answers (list): the list of final answers
    """
    
    if args.answers_are_available:
        questions = []
        rationales = []
        final_answers = []
        decoder = json.JSONDecoder()

        if args.dataset == "gsm8k":
            with open(args.data_path) as f:
                lines = f.readlines()
                for line in lines:
                    json_res = decoder.raw_decode(line)[0]
                    questions.append(f"Q: {json_res['question'].strip()}")
                    rationales.append(f"A: Let's think step by step.\n{json_res['answer'].split('####')[0].strip()}")
                    final_answers.append(json_res["answer"].split("#### ")[-1].replace(",", ""))

        elif args.dataset == "aqua":
            with open(args.data_path) as f:
                lines = f.readlines()
                for line in lines:
                    json_res = decoder.raw_decode(line)[0]
                    qes = json_res["question"].strip() + " Answer Choices:"

                    for opt in json_res["options"]:
                        opt = opt.replace(')', ') ')
                        qes += f" ({opt}"

                    questions.append(f'Q: {qes}')
                    rationales.append(f"A: Let's think step by step.\n{json_res['rationale']}")
                    final_answers.append(json_res["correct"])
        else:
            raise NotImplementedError

        print(f"Dataset: {args.dataset}")
        print(f"Original Dataset size: {len(final_answers)}")
        return questions, rationales, final_answers

    else:
        questions = []
        decoder = json.JSONDecoder()

        if args.dataset == "gsm8k":
            with open(args.data_path) as f:
                lines = f.readlines()
                for line in lines:
                    json_res = decoder.raw_decode(line)[0]
                    questions.append(f"Q: {json_res['question'].strip()}")

        elif args.dataset == "aqua":
            with open(args.data_path) as f:
                lines = f.readlines()
                for line in lines:
                    json_res = decoder.raw_decode(line)[0]
                    qes = json_res["question"].strip() + " Answer Choices:"

                    for opt in json_res["options"]:
                        opt = opt.replace(')', ') ')
                        qes += f" ({opt}"

                    questions.append(f'Q: {qes}')
        else:
            raise NotImplementedError

        print(f"dataset: {args.dataset}")
        print(f"dataset_size: {len(questions)}")
        return questions
        
# return a customized dataloader of batches
# Not PyTorch dataloader, it supprts random index(slice) access
def create_dataloader(args) -> list:
    """
        Create a dataloader for the given dataset
        Args:
            args (object): the arguments passed in from the command line
        Returns:
            dataset (list): the list of questions, rationales and final answers
    """
    dataset = []
    if args.answers_are_available:

        questions, rationales, answers = load_data(args)
        for idx in range(len(questions)):
            dataset.append({"question":questions[idx], "rationale" : rationales[idx],
                            "final_answer":answers[idx]})
    else:
        questions = load_data(args)
        for idx in range(len(questions)):
            dataset.append({"question":questions[idx]})

    random.shuffle(dataset)
    return dataset


def create_several_input_prompts(args: object, cot_flag:bool) -> list:
    """
        Create a list of prompts for the given dataset
        Args:
            args (object): the arguments passed in from the command line
        Returns:
            prompt_demos (list): the list of prompts
    """
    prompt_demos = []
    for prompt_filename in os.listdir(args.dir_prompts):
        prompt = create_single_input_prompt(args, prompt_filename, cot_flag=cot_flag)
        prompt_demos.append(prompt)
    return prompt_demos

# read the generated/prepared prompt json file
# return a string of prefix prompt before each question
def create_single_input_prompt(args: object, prompt_filename: str, cot_flag:bool)->str:
    """
        Create a prompt for the given dataset
        Args:
            args (object): the arguments passed in from the command line
            prompt_filename (str): the name of the prompt file
            cot_flag (bool): whether to use chain of thought or not
        Returns:
            full prompt (str)
    """
    x, z, y = [], [], []
    
    prompt_path = os.path.join(args.dir_prompts, prompt_filename)
    with open(prompt_path, encoding="utf-8") as f:
        json_data = json.load(f)
        json_data = json_data["demo"]
        for line in json_data:
            x.append(line["question"])
            z.append(line["rationale"])
            y.append(line["final_answer"]) # use final_answer instead of pred_ans

    index_list = list(range(len(x)))
    
    prompt_text = ""
    for i in index_list:
        if cot_flag:
            prompt_text += x[i] + "\n" + z[i] + " " + \
                        args.direct_answer_trigger_for_fewshot + " " + y[i] + ".\n\n"
        else:
            prompt_text += x[i] + "\n" + "A: " + args.direct_answer_trigger_for_fewshot + " " + y[i] + ".\n\n"
    return args.prefix + prompt_text + "Q: " + "{question}" + "\nA: Let's think step by step."

def answer_extraction(args: object, responses: str):
    """
        Extract the answer from the response
        Args:
            args (object): the arguments passed in from the command line
            responses (str): the response from the model
        Returns:
            answer (str): the answer extracted from the response
    """
    pred_ans = ""
    temp = responses

    if args.dataset in ("gsm8k"):
        temp = temp.replace(",", "")
        temp = [s for s in re.findall(r'-?\d+\.?\d*', temp)]
    elif args.dataset in ("aqua"):
        if 'none of the answer choices are correct' in responses.lower() or 'The answer is not given in the answer choices' in responses.lower() or 'The answer is not given in the answer choices' in responses.lower() or "The answer is not listed in the answer choices" in responses.lower():
            return 'No answer'
        temp = re.findall(r'A|B|C|D|E', temp)
    
    if len(temp) != 0:
        answer = temp[-1]
        # if there is . at the end of answer, remove it
        # e.g. answer = 64.
        if answer != "":
            if answer[-1] == ".":
                answer = answer[:-1]

        # round the answer to nearest integer
        if args.dataset in ("gsm8k"):
            try:
                answer = str(round(float(answer)))
            except:
                answer = "" # no sol or sol doesn't have valid format
        pred_ans = answer
    else:
        pred_ans = ""

    return pred_ans


def find_most_frequent(arr, n):
    arr_acounts = Counter(arr[:n])
    most_frequent_item, frequency = arr_acounts.most_common(1)[0]
    return frequency, most_frequent_item


def run_llm_extract_answer(args, question):
    response = args.llm_chain.run(question=question)
    return answer_extraction(args, response), response

def single_question_inference(args: object, example, example_idx, correct_count_single_run, wrong_single_run, QA_record_single_run):
    all_self_consistency_ans = []

    QA_record = []
    # enable self-consistency if multipath > 1
    for _ in range(0, args.multipath):
        pred_ans, response = run_llm_extract_answer(args, example['question'])

        # create a dict to record each Q&A for later review purposes
        QA = {}
        QA['question_idx'] = example_idx
        QA['Question'] = example['question']
        QA['Pred_Rationale'] = response
        QA['Pred_FinalAnswer'] = pred_ans
        QA['True_FinalAnswer'] = example['final_answer']
        QA_record.append(QA)

        # output current inference result (only works when self-consistency is not enable)
        if args.multipath == 1:
            print('-' * 20)
            print(f'Question Nr: {example_idx}')
            print(f"Question: \n" + example['question'])
            print(f"Rationale: \nLet's think step by step. " + response)
            print(f"Prediction: {pred_ans}")
            print(f"Ground Truth: {example['final_answer']}")

        # record all answers into the self-consistency list to find the most frequent one
        all_self_consistency_ans.append(pred_ans)

    final_consistent_ans = find_most_frequent(all_self_consistency_ans, args.multipath)[-1]

    if final_consistent_ans == example['final_answer']:
        correct_count_single_run += 1
    else:
        wrong_single_run.append({'idx': example_idx, 'pred_final_answer':final_consistent_ans, 'true_final_answer':example['final_answer']})

    QA_record_single_run.append(QA_record)
    return correct_count_single_run, wrong_single_run, QA_record_single_run

def single_run_inference(data_loader, args):
    correct_count_single_run = 0
    wrong_single_run = [{'prompt' : args.llm_chain.prompt.template}]
    QA_record_single_run = [{'prompt': args.llm_chain.prompt.template}]
    for example_idx, example in enumerate(data_loader):
        correct_count_single_run, wrong_single_run, QA_record_single_run = single_question_inference(args, example, example_idx, correct_count_single_run, wrong_single_run, QA_record_single_run)
    
    return correct_count_single_run, wrong_single_run, QA_record_single_run
        

def all_prompts_inference(args, data_loader, prompts_list):
    all_prompts_correct_count_list = []
    all_prompts_wrong_list = []
    all_prompts_QA_record_list = []
    for i in range(len(prompts_list)):
        args.llm_chain = initialize_llmchain(prompts_list[i], args)
        print(f'PROMPT:\n{args.llm_chain.prompt.template}\n')
        print('START INFERENCE\n')
        #print('*' * 60)
        #continue 
        correct, wrong, QA_record = single_run_inference(data_loader, args)
        all_prompts_correct_count_list.append(correct)
        all_prompts_wrong_list.append(wrong)
        all_prompts_QA_record_list.append(QA_record)
        print('-' * 60)

    #sys.exit(0)
    return all_prompts_correct_count_list, all_prompts_wrong_list, all_prompts_QA_record_list


def generate_uncertainty_single_question(args, example):

    if args.dataset == "gsm8k":
        # the float is reserved for variance calculation result
        if args.answers_are_available:
            uncertainty_record = {'question': example['question'],
                                 'rationale': example['rationale'], 'final_answer': example['final_answer'] , 
                                 'variance':float, 'entropy':float, 'occurrence':{}}
        else:
            uncertainty_record = {'question': example['question'],
                                  'variance':float, 'entropy':float, 'occurrence':{}}
    else:
        if args.answers_are_available:
            uncertainty_record = {'question': example['question'],
                                'rationale': example['rationale'], 'final_answer': example['final_answer'],
                                'entropy':float, 'occurrence':{}}
        else:
            uncertainty_record = {'question': example['question'],
                                  'entropy':float, 'occurrence':{}}

    for _ in range(args.num_trails):
        pred_ans, _ = run_llm_extract_answer(args, example['question'])

        #print(f'Single Trial Rationale:\n{response}')
        print(f'Single Trial Final Answer: {pred_ans}\n')

        # check uncertainty
        if pred_ans != "":
            if pred_ans in uncertainty_record['occurrence']:
                uncertainty_record['occurrence'][pred_ans] += 1 # increment answer occurrence
            else:
                uncertainty_record['occurrence'][pred_ans] = 1 # first occurence
        else:
            # Handle no solution case
            if NO_SOLUTION in uncertainty_record['occurrence']:
                uncertainty_record['occurrence'][NO_SOLUTION] += 1
            else:
                uncertainty_record['occurrence'][NO_SOLUTION] = 1

    # calculate the variance for the question (only applied to datasets with numerical answer)
    if args.dataset == "gsm8k":
        ans_list = []
        for ans, occurs in uncertainty_record['occurrence'].items():
            for i in range(int(occurs)):
                ans_list.append(float(ans))
        uncertainty_record['variance'] = np.var(ans_list)
        
    # calculate the entropy for all dataset
    frequency_list = list(uncertainty_record['occurrence'].values())
    uncertainty_record['entropy'] = entropy(frequency_list)

    # calculate the disagreement for all dataset
    uncertainty_record['disagreement'] = len(uncertainty_record['occurrence'])
    
    return uncertainty_record


# return a sorted list by uncertainty from high to low
def generate_uncertainty_all_questions(args, dataloader):
    result = []
    for example in dataloader:
        print(f'Question: {example["question"]}\n')
        uncertainty_record = generate_uncertainty_single_question(args, example)
        print(f'Uncertainty Record: {uncertainty_record}')
        result.append(uncertainty_record)
        print('\n' + '*' * 60 + '\n')
    
    if args.sort_by == "disagreement":
        result.sort(key=lambda x: -len(x['occurrence']))
    elif args.sort_by == "variance" and args.dataset == "gsm8k":
        result.sort(key=lambda x: -x['variance'])
    elif args.sort_by == "entropy" :
        result.sort(key=lambda x:-x['entropy'])
    return result

def inference_save_info(args, correct_list, wrong_list, QA_record_list, prompts_list, len_dataloader):
    acc_prompt_list = []
    if args.output_dir is not None:
        for i in range(len(correct_list)):
            if prompts_list:
                acc_prompt_dic = {'prompt' : prompts_list[i],
                                'accuracy': correct_list[i] / len_dataloader}
            else:
                acc_prompt_dic = {'accuracy': correct_list[i] / len_dataloader}

            acc_prompt_list.append(acc_prompt_dic)

            wrong = wrong_list[i]
            QA_record = QA_record_list[i]
            path = f"{args.output_dir}wrong_prompt{i+1}.txt"
            orginal_stdout = sys.stdout
            with open(path, 'w', encoding='utf-8') as f:
                sys.stdout = f
                for j in wrong:
                    print(str(j))
            sys.stdout = orginal_stdout

            path = f"{args.output_dir}QA_record_prompt{i+1}.txt"
            with open(path, 'w', encoding='utf-8') as f:
                f.write(json.dumps(QA_record, indent=4))

        overall_mean = np.mean([dic['accuracy'] for dic in acc_prompt_list])
        acc_prompt_list.append({'mean_accuracy': overall_mean})
        path = f"{args.output_dir}accuracy_prompts.txt"
        with open(path, 'w') as f:
            f.write(json.dumps(acc_prompt_list, indent=4))