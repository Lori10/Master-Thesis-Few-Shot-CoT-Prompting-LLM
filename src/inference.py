from utils import *
import time
import argparse
import sys
import json
from utils import initialize_llmchain 
import sys
from constant_vars import *
import datetime

def main():
    # load arguments from terminal
    args = arg_parser()

    current_time = datetime.datetime.now()
    time_string = current_time.strftime("%Y_%m_%d_%H_%M_%S")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        os.makedirs(args.output_dir + '/' + time_string)
    else:
        os.makedirs(args.output_dir + '/' + time_string)

    args.output_dir = args.output_dir + '/' + time_string + '/'
        
    set_random_seed(args.random_seed)
    # load dataset
    dataloader = create_dataloader(args)
    if args.dataset_size_limit <= 0:
        args.dataset_size_limit = len(dataloader)
    else:
        dataloader = dataloader[:args.dataset_size_limit] # replace 7 with 1000; only take 1000 questions randomly to annotate, randomness decided by seed
    
    print('Hyperparameters:')
    print(f'Dataset: {args.dataset}')
    print(f"Dataloader size: {len(dataloader)}")
    print(f'Method: {args.method}')
    print(f'Model: {args.model_id}')
    print(f'Multipath: {args.multipath}')
    print(f'Temperature: {args.temperature}')

    if args.dataset == "gsm8k":
        prefix = prefix_gsm8k
    elif args.dataset == "aqua":
        prefix = prefix_aqua
    else:
        raise NotImplementedError("dataset not implemented")

    if args.method == 'zero_shot_cot':
        if args.dataset == 'aqua':
            prefix = prefix + ' If none of options is correct, please choose the option "None of the above".'
        input_prompt_list = [prefix + "\nQ: " + "{question}" + "\nA: Let's think step by step."]
    elif args.method == 'cot':
        args.prefix = prefix + ' To generate the answer follow the format of the examples below:\n'
        if args.prompt_is_built:
            input_prompt_list = []
            for file_path in os.listdir(args.dir_prompts):
                prompt_full_path = os.path.join(args.dir_prompts, file_path)
                with open(prompt_full_path, 'r', encoding='utf-8') as file:
                    input_prompt_list.append(args.prefix + file.read() + "\nQ: " + "{question}" + "\nA: Let's think step by step.")
        else:
            input_prompt_list = create_several_input_prompts(args, cot_flag=True)
    elif args.method == 'standard':
        args.prefix = prefix + '\n'
        input_prompt_list = create_several_input_prompts(args, cot_flag=False)

    if args.multipath != 1:
        print("Self-consistency Enabled, output each inference result is not available")
    
    start = time.time()
    correct_list, wrong_list, QA_record_list = inference_cot(args, dataloader, input_prompt_list)
    end = time.time()
    print(f"Execution time: {end - start} seconds")
    assert len(correct_list) == len(wrong_list) == len(QA_record_list)
    

    acc_prompt_list = []
    if args.output_dir is not None:
        for i in range(len(correct_list)):
            acc_prompt_dic = {'prompt' : input_prompt_list[i],
                              'accuracy': correct_list[i] / len(dataloader)}
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

        args_dict = {
                    "dataset": args.dataset,
                    "dataset_size_limit": len(dataloader),
                    "data_path": args.data_path,
                    "random_seed": args.random_seed,
                    "dir_prompts": args.dir_prompts,
                    "model_id": args.model_id,
                    "method": args.method,
                    "output_dir": args.output_dir,
                    "temperature": args.temperature,
                    "multipath": args.multipath,
                    "answers_are_available": args.answers_are_available,
                    "prompt_is_built": args.prompt_is_built,
                }
        with open(args.output_dir + 'inference_args.json', 'w') as f:
            json.dump(args_dict, f, indent=4)

def single_run_inference(llm_chain, question_pool, args):
    correct_count = 0
    wrong = [{'prompt' : llm_chain.prompt.template}]
    QA_record = [{'prompt': llm_chain.prompt.template}]
    
    for qes_nr, qes in enumerate(question_pool):
        # create a list for each question to record all answers generated from self-consistency
        all_self_consistency_ans = []

        # enable self-consistency if multipath > 1
        for _ in range(0, args.multipath):
            response = llm_chain.run(question=qes['question'])

            pred_ans = answer_extraction(args, response)

            # create a dict to record each Q&A for later review purposes
            QA = {}
            QA['qes_idx'] = qes_nr
            QA['Q'] = qes['question']
            QA['Pred_Rationale'] = response
            QA['Pred_FinalAnswer'] = pred_ans
            QA['True_FinalAnswer'] = qes['final_answer']
            QA_record.append(QA)

            # output current inference result (only works when self-consistency is not enable)
            if args.multipath == 1:
                print('-' * 20)
                print(f'Question Nr: {qes_nr}')
                print(f"Question: \n" + qes['question'])
                print(f"Rationale: \nLet's think step by step. " + response)
                print(f"Prediction: {pred_ans}")
                print(f"Ground Truth: {qes['final_answer']}")

            # record all answers into the self-consistency list to find the most frequent one
            all_self_consistency_ans.append(pred_ans)

        final_consistent_ans = find_most_frequent(all_self_consistency_ans, args.multipath)[-1]

        if final_consistent_ans == qes['final_answer']:
            correct_count += 1
        else:
            wrong.append({'idx': qes_nr, 'pred_final_answer':final_consistent_ans, 'true_final_answer':qes['final_answer']})

    return correct_count, wrong, QA_record

def inference_cot(args, question_pool, given_prompt):
    correct_count_list = []
    wrong_list = []
    QA_record_list = []
    for i in range(len(given_prompt)):
        llm_chain = initialize_llmchain(given_prompt[i], args)
        print(f'PROMPT:\n{llm_chain.prompt.template}\n')
        print('START INFERENCE\n')
        #print('*' * 60)
        #continue 

        correct, wrong, QA_record = single_run_inference(llm_chain, question_pool, args)
        correct_count_list.append(correct)
        wrong_list.append(wrong)
        QA_record_list.append(QA_record)
        print('-' * 60)

    #sys.exit(0)
    return correct_count_list, wrong_list, QA_record_list
    
        
def arg_parser():
    parser = argparse.ArgumentParser(description="CoT")
    parser.add_argument("--random_seed", type=int, default=1, help="random seed")
    parser.add_argument(
        "--dataset", type=str, default="gsm8k", choices=["gsm8k", "aqua"], help="dataset to inference"
    )

    parser.add_argument(
        "--data_path", type=str, default="../datasets/gsm8k/test.jsonl", choices=["../datasets/AQuA/test.json", "../datasets/gsm8k/test.jsonl"], help="dataset to inference"
    )

    parser.add_argument(
        "--dir_prompts", type=str, default="labeled_demos/active/gsm8k/model_text-davinci-003_method_few_shot_cot_numtrails_3_sortby_entropy_temperature_0-7_seed_42_nrdemos_3_datasetsizelimit_5", help="prompts to use"
    )
    parser.add_argument(
        "--model_id", type=str, default="text-davinci-003", choices=["text-davinci-003", "tiiuae/falcon-7b-instruct"], help="model used for decoding."
    )

    parser.add_argument(
        "--method", type=str, default="cot", choices=["zero_shot_cot", "standard", "cot"], help="method"
    )

    parser.add_argument(
        "--output_dir", type=str, default="inference_results", help="output directory"
    )
    
    parser.add_argument(
        "--dataset_size_limit", type=int, default=5, help="whether to limit the dataset size. if 0, the dataset size is unlimited and we use all the samples in the dataset for creating the demonstrations."
    )
  
    parser.add_argument(
        "--temperature", type=float, default=0, help="temperature used for llm decoding"
    )
    parser.add_argument(
        "--multipath", type=int, default=1, help="self-consistency path num"
    )
   
    parser.add_argument(
        "--answers_are_available", type=bool, default=True, help='true if answers are available in the test dataset, false otherwise'
    )

    parser.add_argument(
        "--prompt_is_built", type=bool, default=False, help="if the prompt is already built as a string, set this to true"
    )

    args = parser.parse_args()

    if args.multipath > 1:
        args.temperature = 0.7
    else:
        args.temperature = 0
    
    if args.dataset == "gsm8k":
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif args.dataset == "aqua":
        args.direct_answer_trigger = "The answer is"

    elif args.dataset == "svamp":
        args.dataset_path = "./dataset/SVAMP/SVAMP.json"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif args.dataset == "asdiv":
        args.dataset_path = "./dataset/ASDiv/ASDiv.json"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif args.dataset == "csqa":
        args.dataset_path = "./dataset/CSQA/dev_rand_split.jsonl"
        args.direct_answer_trigger = "So the answer is"
    elif args.dataset == "strategyqa":
        args.dataset_path = "./dataset/strategyQA/task.json"
        args.direct_answer_trigger = "\nTherefore, the answer (Yes or No) is"
    elif args.dataset == "last_letters":
        args.dataset_path = "./dataset/last_letters/last_letters_test.json"
        args.direct_answer_trigger = "\nTherefore, the answer is"
    elif args.dataset == "addsub":
        args.dataset_path = "./dataset/MAWPS/AddSub.json"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif args.dataset == "singleeq":
        args.dataset_path = "./dataset/MAWPS/SingleEq.json"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif args.dataset == "multiarith":
        args.dataset_path = "./dataset/MAWPS/MultiArith.json"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    else:
        raise ValueError("dataset is not properly defined ...")
        
    trigger = args.direct_answer_trigger.replace("\nTherefore, ", "")
    args.direct_answer_trigger_for_zeroshot = trigger[0].upper() + trigger[1:]
    args.direct_answer_trigger_for_zeroshot_cot = args.direct_answer_trigger
    args.direct_answer_trigger_for_fewshot = "The answer is"
    args.cot_trigger = "Let's think step by step."
    
    return args


if __name__ == "__main__":
    main()