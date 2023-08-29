from utils.final_answer_extraction import run_llm_extract_answer 
import numpy as np
from scipy.stats import entropy
from constant_vars import NO_SOLUTION
from utils.prompts_llm import initialize_llmchain, create_header_llm


def sort_uncertainty(args, result):
    if args.sort_by == "disagreement":
        result.sort(key=lambda x: -len(x['occurrence']))
    elif args.sort_by == "variance" and args.dataset == "gsm8k":
        result.sort(key=lambda x: -x['variance'])
    elif args.sort_by == "entropy" :
        result.sort(key=lambda x:-x['entropy'])
    return result

def generate_uncertainty_single_question(args, example, azure_llm_chain, openai_llm_chain):

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

    nr_preds_openai = {
                     'question_idx' : example['question_idx'], 
                     'nr_preds': 0
                     }
    for _ in range(args.num_trails):
        try:
            pred_ans, _ = run_llm_extract_answer(args, azure_llm_chain, example['question'])

        except Exception as e_azure:
            print(f'For this question, Error Generated when using AzureChatOpenAI: {e_azure}. Proceeding to ChatOpenAI.')
            try:
                pred_ans, _ = run_llm_extract_answer(args, openai_llm_chain, example['question'])
                nr_preds_openai['nr_preds'] += 1

            except Exception as e_openai:
                print(f'For this question, Error Generated when using ChatOpenAI: {e_openai}. Stopping the program.')

                raise Exception(e_openai)
                

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
            for _ in range(int(occurs)):
                ans_list.append(float(ans))
        uncertainty_record['variance'] = np.var(ans_list)
        
    # calculate the entropy for all dataset
    frequency_list = list(uncertainty_record['occurrence'].values())
    uncertainty_record['entropy'] = entropy(frequency_list)

    # calculate the disagreement for all dataset
    uncertainty_record['disagreement'] = len(uncertainty_record['occurrence'])
    
    return uncertainty_record, nr_preds_openai

def generate_uncertainty_all_questions(args, dataloader, sort, azure_llm_chain, openai_llm_chain):
    try:
        result = []
        nr_answers_openai_list = []
        for example_id, example in enumerate(dataloader):
            print(f'Example ID: {example_id}')
            print(f'Question:\n{example["question"]}\n')

            uncertainty_record, nr_answer_openai_single_example = generate_uncertainty_single_question(args, example, azure_llm_chain, openai_llm_chain)

            if nr_answer_openai_single_example['nr_preds'] > 0:
                nr_answers_openai_list.append(nr_answer_openai_single_example)

            print(f'Uncertainty Record: {uncertainty_record}')
            result.append(uncertainty_record)
            print('\n' + '*' * 60 + '\n')

        if sort:
            return sort_uncertainty(args, result)
        else:
            return result, nr_answers_openai_list
            
    except Exception as e:
        print(f'Error occured during uncertainty estimation! Error message: {e}')
        return result, nr_answers_openai_list