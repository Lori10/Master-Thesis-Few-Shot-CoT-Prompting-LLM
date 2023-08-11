import random 
import numpy as np
from typing import Tuple
import json

def set_random_seed(seed: int):
    """
        Fix the seed for random and numpy
        Args:
            seed (int): the seed to fix
    """
    # random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    
    
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

    set_random_seed(args.random_seed)
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
    # add index to each example
    for idx, example in enumerate(dataset):
        example['question_idx'] = idx

    if args.dataset_size_limit <= 0:
        args.dataset_size_limit = len(dataset)
    else:
        dataset = dataset[:args.dataset_size_limit]

    return dataset