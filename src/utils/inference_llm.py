from utils.final_answer_extraction import run_llm_extract_answer, find_most_frequent
from utils.prompts_llm import initialize_llmchain, from_chatmodelmessages_to_string
import sys

def single_question_inference(args: object, example, example_idx, correct_count_single_run, wrong_single_run, QA_record_single_run, azure_llm_chain, openai_llm_chain):
    all_self_consistency_ans = []

    QA_record = []

    is_answer_from_openai = False
    # enable self-consistency if multipath > 1
    for _ in range(0, args.multipath):
        try:
            pred_ans, response = run_llm_extract_answer(args, azure_llm_chain, example['question'])

        except Exception as e_azure:
            print(f'For this question, Error Generated when using AzureChatOpenAI: {e_azure}. Proceeding to ChatOpenAI.')
            try:
                pred_ans, response = run_llm_extract_answer(args, openai_llm_chain, example['question'])
                is_answer_from_openai = True
            except Exception as e_openai:
                print(f'For this question, Error Generated when using ChatOpenAI: {e_openai}. Stopping the program.')

                raise Exception(e_openai)

        # create a dict to record each Q&A for later review purposes
        QA = {}
        QA['Question_idx'] = example_idx
        QA['Question'] = example['question']
        QA['Pred_Rationale'] = response
        QA['Pred_FinalAnswer'] = pred_ans
        QA['True_FinalAnswer'] = example['final_answer']
        QA_record.append(QA)

        # output current inference result (only works when self-consistency is not enable)
        if args.multipath == 1:
            print('-' * 20)
            print(f'Question idx: {example_idx}')
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
        wrong_single_run.append({'question_idx': example_idx, 'pred_final_answer':final_consistent_ans, 'true_final_answer':example['final_answer']})

    QA_record_single_run.append(QA_record)
    return correct_count_single_run, wrong_single_run, QA_record_single_run, is_answer_from_openai

def single_run_inference(data_loader, args, azure_llm_chain, openai_llm_chain):
    prompt = from_chatmodelmessages_to_string(azure_llm_chain.prompt.messages) if args.model_id.startswith("gpt-35") else args.azure_llm_chain.prompt.template
    print(f'PROMPT TEMPLATE:\n{prompt}\n')
    print('START INFERENCE\n')
    
    correct_count_single_run = 0
    wrong_single_run = [{'prompt' : prompt}]
    QA_record_single_run = [{'prompt': prompt}]
    is_answer_openai_info = {'prompt': prompt, 
                             'is_answer_openai': 0}
    
    for example_idx, example in enumerate(data_loader):
        correct_count_single_run, wrong_single_run, QA_record_single_run, is_answer_openai = single_question_inference(args, example, example_idx, correct_count_single_run, wrong_single_run, QA_record_single_run, azure_llm_chain, openai_llm_chain)
        if is_answer_openai:
            is_answer_openai_info['is_answer_openai'].append(example_idx)

    return correct_count_single_run, wrong_single_run, QA_record_single_run, is_answer_openai_info

def all_prompts_inference(args, data_loader, prompts_list, azure_llm, openai_llm):    
    all_prompts_correct_count_list = []
    all_prompts_wrong_list = []
    all_prompts_QA_record_list = []
    list_answers_openai = []
    for _ in range(len(prompts_list)):
        azure_llm_chain = initialize_llmchain(azure_llm, prompts_list[0])
        openai_llm_chain = initialize_llmchain(openai_llm, prompts_list[0])  
        
        correct, wrong, QA_record, is_answer_openai  = single_run_inference(data_loader, args, azure_llm_chain, openai_llm_chain)
            
        all_prompts_correct_count_list.append(correct)
        all_prompts_wrong_list.append(wrong)
        all_prompts_QA_record_list.append(QA_record)
        list_answers_openai.append(is_answer_openai)
        print('-' * 60)

    #sys.exit(0)
    return all_prompts_correct_count_list, all_prompts_wrong_list, all_prompts_QA_record_list, list_answers_openai


