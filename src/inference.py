from utils import *
import time
import argparse
import sys
import json
from generate_demo_active import predict_llm 
import sys

def main():
    # load arguments from terminal
    args = arg_parser()
    print('*****************************')
    print(args)
    print('*****************************')

    print(f"API_KEY: {API_KEY}")

    set_random_seed(args.random_seed)

    # load dataset
    dataloader = create_dataloader(args)

    if args.method == "standard":
        input_prompt_list = create_several_input_prompts(args, cot_flag=False)
    elif args.method == "random_cot" or args.method == "auto_cot" or args.method == "active_cot":
        input_prompt_list = create_several_input_prompts(args, cot_flag=True)
    else:
        raise NotImplementedError

    start = time.time()
    print("Inference Start")
    if args.multipath != 1:
        print("Self-consistency Enabled, output each inference result is not available")
    # no limit on how many batches to inference, assume inference all batches
    if args.qes_limit == 0:
        dataloader = dataloader[:15] # only take 1000 questions randomly to annotate, randomness decided by seed
        args.qes_limit = len(dataloader)
    
    correct_list, wrong_list, QA_record_list = inference_cot(args, dataloader, args.qes_limit, input_prompt_list)
    end = time.time()
    print(f"Execution time: {end - start} seconds")
    assert len(correct_list) == len(wrong_list) == len(QA_record_list)
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        os.makedirs(args.output_dir + args.method)
        os.makedirs(args.output_dir + args.method + '/' + args.dataset)
    elif not os.path.exists(args.output_dir + args.method):
        os.makedirs(args.output_dir + args.method)
        os.makedirs(args.output_dir + args.method + '/' + args.dataset)
    elif not os.path.exists(args.output_dir + args.method  + '/' + args.dataset):
        os.makedirs(args.output_dir + args.method + '/' + args.dataset)

    args.output_dir = f"{args.output_dir}/{args.method}/{args.dataset}/"

    acc_prompt_list = []
    if args.output_dir is not None:
        for i in range(len(correct_list)):
            acc_prompt_dic = {'prompt' : input_prompt_list[i],
                              'accuracy': correct_list[i] / args.qes_limit}
            acc_prompt_list.append(acc_prompt_dic)

            wrong = wrong_list[i]
            QA_record = QA_record_list[i]
            path = f"{args.output_dir}wrong_prompt{i+1}.txt"
            orginal_stdout = sys.stdout
            with open(path, 'w') as f:
                sys.stdout = f
                for j in wrong:
                    print(str(j))
            sys.stdout = orginal_stdout

            path = f"{args.output_dir}QA_record_prompt{i+1}.txt"
            with open(path, 'w') as f:
                f.write(json.dumps(QA_record, indent=4))

        overall_mean = np.mean([dic['accuracy'] for dic in acc_prompt_list])
        acc_prompt_list.append({'mean_accuracy': overall_mean})
        path = f"{args.output_dir}accuracy_prompts.txt"
        with open(path, 'w') as f:
            f.write(json.dumps(acc_prompt_list, indent=4))

    
def single_run_inference(single_prompt, question_pool, qes_limit, args):
    correct_count = 0
    qes_count = 0
    wrong = [{'prompt' : single_prompt}]
    QA_record = [{'prompt': single_prompt}]
    
    for qes_num, qes in enumerate(question_pool):
        if qes_limit is not None and qes_count == qes_limit:
            break
        # create a list for each question to record all answers generated from self-consistency
        all_self_consistency_ans = []

        if args.dataset == "last_letters" and args.use_code_style_prompt is True:
            # code style prompt
            prompt = single_prompt + "Q: " + "{question}" + "\nA: Let's think step by step in Python."
        else:
            prompt = single_prompt + "Q: " + "{question}" + "\nA: Let's think step by step."
        
        # print(f'PROMPT: {prompt}')
        # sys.exit(0)

        # enable self-consistency if multipath > 1
        for path in range(0, args.multipath):

            responses, _, _ = predict_llm(template=prompt, question=qes['question'], model="gpt-3.5-turbo",
                                      max_tokens=args.max_length_cot, time_interval=args.api_time_interval, 
                                      stop='\n', temperature=args.temperature)
            pred_ans = answer_extraction(args, responses)

            # create a dict to record each Q&A for later review purposes
            QA = {}
            QA['qes_idx'] = qes['question_idx']
            QA['Q'] = qes['question']
            #QA['A'] = responses['choices'][0]['text']
            #QA['A'] = responses.choices[0].message.content
            QA['Pred_Rationale'] = responses
            QA['Pred_FinalAnswer'] = pred_ans
            QA_record.append(QA)

            # output current inference result (only works when self-consistency is not enable)
            if args.multipath == 1:
                print('-' * 20)
                print(f"Question number: {qes_num}")
                print(f"Dataset index: {qes['question_idx']}")
                print(f"Question: \n" + qes['question'])
                if args.dataset == "last_letters" and args.use_code_style_prompt is True:
                    #print(f"A: Let's think step by step in Python." + responses['choices'][0]['text'])
                    #print(f"A: Let's think step by step in Python." + responses.choices[0].message.content)
                    print(f"Let's think step by step in Python.\n" + responses)

                else:
                    #print(f"A: Let's think step by step." + responses['choices'][0]['text'])
                    #print(f"A: Let's think step by step in Python." + responses.choices[0].message.content)
                    print(f"Let's think step by step.\n" + responses)


                print(f"Prediction: {pred_ans}")
                print(f"Ground Truth: {qes['final_answer']}")

            # record all answers into the self-consistency list to find the most frequent one
            all_self_consistency_ans.append(pred_ans)

        final_consistent_ans = find_most_frequent(all_self_consistency_ans, args.multipath)[-1]

        if final_consistent_ans == qes['final_answer']:
            correct_count += 1
        else:
            wrong.append({'idx':qes['question_idx'], 'pred_final_answer':final_consistent_ans, 'true_final_answer':qes['final_answer']})

        qes_count += 1

    return correct_count, wrong, QA_record

def inference_cot(args, question_pool, qes_limit, given_prompt):
    correct_count_list = []
    wrong_list = []
    QA_record_list = []
    for i in range(len(given_prompt)):
        correct, wrong, QA_record = single_run_inference(given_prompt[i], question_pool, qes_limit, args)
        correct_count_list.append(correct)
        wrong_list.append(wrong)
        QA_record_list.append(QA_record)

    return correct_count_list, wrong_list, QA_record_list
    
        

def arg_parser():
    parser = argparse.ArgumentParser(description="CoT")
    parser.add_argument("--random_seed", type=int, default=1, help="random seed")
    parser.add_argument(
        "--dataset", type=str, default="aqua", choices=["gsm8k","svamp", "aqua", "csqa", "asdiv", "last_letters", "addsub", "singleeq", "strategyqa", "multiarith"], help="dataset to inference"
    )
    parser.add_argument(
        "--dir_prompts", type=str, default="demos/random/aqua", help="prompts to use"
    )
    parser.add_argument(
        "--model", type=str, default="text-davinci-002", choices=["text-davinci-002", "code-davinci-002"], help="model used for decoding."
    )
    parser.add_argument(
        "--method", type=str, default="standard", choices=["zero_shot", "zero_shot_cot", "standard", "random_cot", "auto_cot", "active_cot"], help="method"
    )
    parser.add_argument(
        "--output_dir", type=str, default="inference_results/", help="output directory"
    )
    parser.add_argument(
        "--max_length_cot", type=int, default=256, help="maximum length of output tokens by model for reasoning extraction"
    )
    parser.add_argument(
        "--qes_limit", type=int, default=0, help="whether to limit test dataset size. if 0, the dataset size is unlimited and we use all the samples in the dataset for testing."
    )
    parser.add_argument(
        "--api_time_interval", type=float, default=1.0, help="how many seconds to sleep between each request"
    )
    parser.add_argument(
        "--temperature", type=float, default=0, help=""
    )
    parser.add_argument(
        "--multipath", type=int, default=1, help="self-consistency path num"
    )
    parser.add_argument(
        "--concat_length", type=int, default=4, help='Used for task last_letters, indicates length of last letter to concat, i.e. Elon Musk -> nk, use concat length of 2'
    )
    parser.add_argument(
        "--use_code_style_prompt", type=bool, default=False, help='Use code-style prompt as mentioned in paper for last_letters dataset'
    )
    
    args = parser.parse_args()

    if args.multipath > 1:
        args.temperature = 0.7
    else:
        args.temperature = 0
    print(f"Temperature: {args.temperature}")
    
    if args.dataset == "gsm8k":
        args.data_path = "../datasets/gsm8k/test.jsonl"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif args.dataset == "aqua":
        args.data_path = "../datasets/AQuA/test.json"
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