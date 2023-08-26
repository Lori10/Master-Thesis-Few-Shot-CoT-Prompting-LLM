from utils.final_answer_extraction import run_llm_extract_answer, find_most_frequent
from utils.prompts_llm import initialize_llmchain, from_chatmodelmessages_to_string
import sys

def single_question_inference(args: object, example, example_idx, correct_count_single_run, wrong_single_run, QA_record_single_run):
    all_self_consistency_ans = []

    QA_record = []
    # enable self-consistency if multipath > 1
    for _ in range(0, args.multipath):
        pred_ans, response = run_llm_extract_answer(args, example['question'])

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
    return correct_count_single_run, wrong_single_run, QA_record_single_run

def single_run_inference(data_loader, args):
    prompt = from_chatmodelmessages_to_string(args.llm_chain.prompt.messages) if args.model_id.startswith("gpt-35") else args.llm_chain.prompt.template
    print(f'PROMPT TEMPLATE:\n{prompt}\n')
    print('START INFERENCE\n')
    
    correct_count_single_run = 0
    wrong_single_run = [{'prompt' : prompt}]
    QA_record_single_run = [{'prompt': prompt}]
    for example_idx, example in enumerate(data_loader):
        correct_count_single_run, wrong_single_run, QA_record_single_run = single_question_inference(args, example, example_idx, correct_count_single_run, wrong_single_run, QA_record_single_run)
    
    return correct_count_single_run, wrong_single_run, QA_record_single_run

def all_prompts_inference(args, data_loader, prompts_list):    
    all_prompts_correct_count_list = []
    all_prompts_wrong_list = []
    all_prompts_QA_record_list = []
    for i in range(len(prompts_list)):
        initialize_llmchain(args, prompts_list[i], llm_init=True)        
        correct, wrong, QA_record = single_run_inference(data_loader, args)
        all_prompts_correct_count_list.append(correct)
        all_prompts_wrong_list.append(wrong)
        all_prompts_QA_record_list.append(QA_record)
        print('-' * 60)

    #sys.exit(0)
    return all_prompts_correct_count_list, all_prompts_wrong_list, all_prompts_QA_record_list


