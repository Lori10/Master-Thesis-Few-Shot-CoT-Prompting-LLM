from src.constant_vars import NO_SOLUTION
from src.utils.final_answer_extraction import run_llm_extract_answer 
import numpy as np
from scipy.stats import entropy

def generate_uncertainty_single_question(args, example):

    if args.dataset == "gsm8k":
        # the float is reserved for variance calculation result
        if args.answers_are_available:
            uncertainty_record = {'question': example['question'], 'question_idx' : example['question_idx'],
                                 'rationale': example['rationale'], 'final_answer': example['final_answer'] , 
                                 'variance':float, 'entropy':float, 'occurrence':{}}
        else:
            uncertainty_record = {'question': example['question'], 'question_idx' : example['question_idx'],
                                  'variance':float, 'entropy':float, 'occurrence':{}}
    else:
        if args.answers_are_available:
            uncertainty_record = {'question': example['question'], 'question_idx': example['question_idx'],
                                'rationale': example['rationale'], 'final_answer': example['final_answer'],
                                'entropy':float, 'occurrence':{}}
        else:
            uncertainty_record = {'question': example['question'], 'question_idx': example['question_idx'],
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

def generate_uncertainty_all_questions(args, dataloader):
    result = []
    for example in dataloader:
        print(f'Question: {example["question"]}\n')
        uncertainty_record = generate_uncertainty_single_question(args, example)
        print(f'Uncertainty Record: {uncertainty_record}')
        result.append(uncertainty_record)
        print('\n' + '*' * 60 + '\n')

    if args.sort:
        if args.sort_by == "disagreement":
            result.sort(key=lambda x: -len(x['occurrence']))
        elif args.sort_by == "variance" and args.dataset == "gsm8k":
            result.sort(key=lambda x: -x['variance'])
        elif args.sort_by == "entropy" :
            result.sort(key=lambda x:-x['entropy'])

    return result