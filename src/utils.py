# This file contains necessary helper functions
# e.g. GPT request, create_dataloader
import openai
import random
import sys
import numpy as np
import torch
import json
import re
from collections import Counter
import time
import os

# put your Openai API_KEY here
API_KEY = ""
# define for no solution if GPT cannot generate a valid solution
# here define a magic number for the convenience of variance calculation
NO_SOLUTION = '-10086'

def fix_seed(seed):
    # random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

# set the random seed for reproducibility
def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# pass in a prompt and returns a response body contains response
def GPT3_request(model:str, input_prompt:list, max_tokens:int, time_interval, temperature=0.7, stop=None):
    resp = None
    done = False
    while not done:
        try:
            openai.api_key = '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'
            resp = openai.Completion.create(
                model=model,
                prompt=input_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                stop = stop
            )
            done = True
        except:
            errno = sys.exc_info()[:2]
            if errno[0] == openai.error.InvalidRequestError:
                print(f"Invalid Request\nPrompt: {input_prompt}\n")
                print(f"Reason: {errno[1]}")
                assert False
            else:
                print(f"Error: {errno[0]}\n")
                print(f"Reason: {errno[1]}\n")
        # pause between each request to avoid rate limit
        time.sleep(time_interval)
    return resp


def ChatGPT3_request(model:str, input_prompt:list, max_tokens:int, time_interval, temperature=0.7, stop=None):
    resp = None
    done = False
    while not done:
        try:
            openai.api_key = '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'
            resp = openai.ChatCompletion.create(
                model='gpt-3.5-turbo',
                messages=[{'role' : 'user', 'content': input_prompt[0]}],
                #prompt=input_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                stop = stop
            )
            done = True
        except:
            errno = sys.exc_info()[:2]
            if errno[0] == openai.error.InvalidRequestError:
                print(f"Invalid Request\nPrompt: {input_prompt[0]}\n")
                print(f"Reason: {errno[1]}")
                assert False
            else:
                print(f"Error: {errno[0]}\n")
                print(f"Reason: {errno[1]}\n")
        # pause between each request to avoid rate limit
        time.sleep(time_interval)
    return resp


def LangChain_model(model:str, input_prompt:list, max_tokens:int, time_interval, temperature=0.7, stop=None):
    resp = None
    done = False
    while not done:
        try:
            openai.api_key = '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'
            resp = openai.Completion.create(
                model=model,
                prompt=input_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                stop = stop
            )
            done = True
        except:
            errno = sys.exc_info()[:2]
            if errno[0] == openai.error.InvalidRequestError:
                print(f"Invalid Request\nPrompt: {input_prompt}\n")
                print(f"Reason: {errno[1]}")
                assert False
            else:
                print(f"Error: {errno[0]}\n")
                print(f"Reason: {errno[1]}\n")
        # pause between each request to avoid rate limit
        time.sleep(time_interval)
    return resp


def load_data(args):
    
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
                    questions.append(f"Q: {json_res['question'].strip()}\nA:")
                    rationales.append(f"Let's think step by step.\n{json_res['answer'].split('####')[0].strip()}")
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

                    questions.append(qes + '\nA:')
                    rationales.append(f"Let's think step by step.\n{json_res['rationale']}")
                    final_answers.append(json_res["correct"])
        else:
            raise NotImplementedError

        print(f"dataset: {args.dataset}")
        print(f"dataset_size: {len(final_answers)}")
        args.dataset_size = len(final_answers)
        return questions, rationales, final_answers

    else:
        questions = []
        decoder = json.JSONDecoder()

        if args.dataset == "gsm8k":
            with open(args.data_path) as f:
                lines = f.readlines()
                for line in lines:
                    json_res = decoder.raw_decode(line)[0]
                    questions.append(f"Q: {json_res['question'].strip()}\nA:")

        elif args.dataset == "aqua":
            with open(args.data_path) as f:
                lines = f.readlines()
                for line in lines:
                    json_res = decoder.raw_decode(line)[0]
                    qes = json_res["question"].strip() + " Answer Choices:"

                    for opt in json_res["options"]:
                        opt = opt.replace(')', ') ')
                        qes += f" ({opt}"

                    questions.append(qes + '\nA:')
        else:
            raise NotImplementedError

        print(f"dataset: {args.dataset}")
        print(f"dataset_size: {len(questions)}")
        return questions
        
# return a customized dataloader of batches
# Not PyTorch dataloader, it supprts random index(slice) access
def create_dataloader(args)->list:
    set_random_seed(args.random_seed)
    dataset = []

    if args.answers_are_available:
        questions, rationales, answers = load_data(args)
        for idx in range(len(questions)):
            dataset.append({"question":questions[idx], "rationale" : rationales[idx],
                            "final_answer":answers[idx], "question_idx":idx})
    else:
        questions = load_data(args)
        for idx in range(len(questions)):
            dataset.append({"question":questions[idx], "question_idx":idx})

    #random.shuffle(dataset)
    return dataset


def create_several_input_prompts(args, cot_flag:bool):
    prompt_demos = []
    for prompt_filename in os.listdir(args.dir_prompts):
        prompt = create_single_input_prompt(args, prompt_filename, cot_flag=cot_flag)
        prompt_demos.append(prompt)
    return prompt_demos

# read the generated/prepared prompt json file
# return a string of prefix prompt before each question
def create_single_input_prompt(args, prompt_filename, cot_flag:bool)->str:
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
            if args.dataset == "strategyqa":
                prompt_text += x[i] + " " + z[i] + " " + \
                            "So the answer is" + " " + y[i] + ".\n\n"
            else:
                prompt_text += x[i] + " " + z[i] + " " + \
                            args.direct_answer_trigger_for_fewshot + " " + y[i] + ".\n\n"
        else:
            prompt_text += x[i] + " " + args.direct_answer_trigger_for_fewshot + " " + y[i] + ".\n\n"
    return prompt_text


def answer_extraction(args, responses):
    pred_ans = ""
    
    #temp = responses['choices'][0].text
    #temp = responses.choices[0].message.content
    temp = responses

    if args.dataset in ("gsm8k", "svamp", "asdiv", "addsub", "singleeq", "multiarith"):
        temp = temp.replace(",", "")
        temp = [s for s in re.findall(r'-?\d+\.?\d*', temp)]
    elif args.dataset in ("aqua", "csqa"):
        if 'none of the answer choices are correct' in responses.lower() or 'The answer is not given in the answer choices' in responses.lower() or 'The answer is not given in the answer choices' in responses.lower() or "The answer is not listed in the answer choices" in responses.lower() or responses == '':
            return 'No answer'
        temp = re.findall(r'A|B|C|D|E', temp)
    elif args.dataset in ("strategyqa", "coin_flip"):
        temp = temp.lower()
        temp = re.sub("\"|\'|\n|\.|\s|\:|\,"," ", temp)
        temp = temp.split(" ")
        temp = [i for i in temp if i in ("yes", "no")]
    elif args.dataset in ("last_letters"):
        temp = re.sub("\"|\'|\n|\.|\s","", temp)
        temp = [temp]
    
    if len(temp) != 0:
        answer = temp[-1]
        # if there is . at the end of answer, remove it
        # e.g. answer = 64.
        if answer != "":
            if answer[-1] == ".":
                answer = answer[:-1]

        # round the answer to nearest integer
        if args.dataset in ("gsm8k", "svamp"):
            try:
                answer = str(round(float(answer)))
            except:
                answer = "" # no sol or sol doesn't have valid format
        elif args.dataset in ("last_letters"):
            try:
                answer = answer[-args.concat_length:]
            except:
                answer = ""
        pred_ans = answer
    else:
        pred_ans = ""

    return pred_ans


def find_most_frequent(arr, n):
    arr_acounts = Counter(arr[:n])
    most_frequent_item, frequency = arr_acounts.most_common(1)[0]
    return frequency, most_frequent_item