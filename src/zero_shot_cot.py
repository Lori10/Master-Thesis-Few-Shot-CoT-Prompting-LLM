import pandas as pd
from tqdm import tqdm
from experiment_llm import predict_llm
from constant_variables import ZERO_SHOT_TEMPLATE_DIC, EXTRACT_ANSWERS_DIC

def zero_shot_cot(dataset_name : str, dataset : dict):
    template = ZERO_SHOT_TEMPLATE_DIC[dataset_name]
    extract_ai_answer_func = EXTRACT_ANSWERS_DIC[dataset_name]['extract_ai_answer_func']
    extract_true_answer_func = EXTRACT_ANSWERS_DIC[dataset_name]['extract_true_answer_func']
    res_df = pd.DataFrame()

    for example in tqdm(dataset[:3]):
        ai_completion, total_tokens, total_cost = predict_llm(template, example['question'], 'gpt-3.5-turbo')
        ai_final_answer = extract_ai_answer_func(ai_completion)
        true_final_answer = extract_true_answer_func(example['answer'])

        row_dic = {'question' : [example['question']],
                   'ai_final_answer' : [ai_final_answer],
                   'ai_completion' : [ai_completion],
                   'true_final_answer' : [true_final_answer],
                   'true_completion' : [example['answer']],
                   'total_tokens' : [total_tokens],
                   'total_price' : [total_cost],
                  }

        res_df = pd.concat([res_df, pd.DataFrame(row_dic)], ignore_index=False)

    #res_df.to_csv(f'../results/Zero_Shot_CoT/{dataset_name}.csv', index=False)
    res_df.to_json(f'../results/Zero_Shot_CoT/{dataset_name}.jsonl', orient='records', lines=True)
    return 'Zero Shot CoT Experiment Run Successfully!' 