import json 
from utils.load_data import create_dataloader
import argparse 
import os
import datetime
from utils.filter_simple_examples import filter_examples_with_labels
from utils.prompts_llm import create_several_input_prompts
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def parse_arguments():
    parser = argparse.ArgumentParser(description="Get-Demo-ZeroShotCoT")
    parser.add_argument(
        "--dataset", type=str, default="gsm8k",
        choices=["aqua", "gsm8k"], help="dataset used for experiment"
    )

    parser.add_argument(
        "--data_path", type=str, default="../datasets/sampled_zeroshotcot_training_data/gsm8k/QA_record_prompt1.txt",
        choices=["../datasets/original/gsm8k/train.jsonl", "../datasets/original/AQuA/train.json",
                 "../datasets/sampled_zeroshotcot_training_data/gsm8k/QA_record_prompt1.txt",
                 "../datasets/sampled_zeroshotcot_training_data/aqua/QA_record_prompt1.txt"], help="dataset used for experiment"
    )

    parser.add_argument("--random_seed", type=int, default=1, help="random seed")

    parser.add_argument(
        "--dataset_size_limit", type=int, default=1000, help="whether to limit training dataset size. if 0, the dataset size is unlimited and we use all the samples in the dataset for creating the demonstrations."
    )

    parser.add_argument(
        "--max_token_len", type=float, default=60, help="maximum number of reasoning chains"
    )

    parser.add_argument(
        "--max_ra_len", type=float, default=5, help="maximum number of reasoning chains"
    )

    parser.add_argument(
        "--answers_are_available", type=bool, default=True, help='true if answers are available in the test dataset, false otherwise'
    )   

    args = parser.parse_args()
    if args.answers_are_available:
        args.demos_save_dir = "labeled_demos"
    else:
        args.max_ra_len = 'None'
        args.demos_save_dir = "unlabeled_demos"
    return args

def main():
    args = parse_arguments()

    # current_time = datetime.datetime.now()
    # time_string = current_time.strftime("%Y_%m_%d_%H_%M_%S")
    # if not os.path.exists(args.demos_save_dir):
    #     os.makedirs(args.demos_save_dir)
    #     os.makedirs(args.demos_save_dir + '/' + 'random')
    #     os.makedirs(args.demos_save_dir + '/' +  'random' + '/' + time_string)
    #     os.makedirs(args.demos_save_dir + '/' + 'random' + '/' + time_string + '/' + 'demos')
    # elif not os.path.exists(args.demos_save_dir + '/' + 'random'):
    #     os.makedirs(args.demos_save_dir + '/' + 'random')
    #     os.makedirs(args.demos_save_dir + '/' +  'random' + '/' + time_string)
    #     os.makedirs(args.demos_save_dir + '/' + 'random' + '/' + time_string + '/' + 'demos')
    # else:
    #     os.makedirs(args.demos_save_dir + '/' +  'random' + '/' + time_string)
    #     os.makedirs(args.demos_save_dir + '/' + 'random' + '/' + time_string + '/' + 'demos')

    # args.demos_save_dir = args.demos_save_dir + '/' + 'random' + '/' + time_string + '/demos'

    
    
    if args.data_path.endswith('json') or args.data_path.endswith('jsonl'):
        dataloader = create_dataloader(args)
    else:

        with open(args.data_path, 'r') as f:
            data = json.load(f)

        dataloader = []
        for entry in data[1:]:
            for question_data in entry:
                question_idx = question_data["Question_idx"]
                pred_final_answer = question_data["Pred_FinalAnswer"]
                true_final_answer = question_data["True_FinalAnswer"]

                data_dic = {}
                data_dic['question_idx'] = question_idx
                data_dic['question'] = question_data['Question']
                data_dic["rationale"] = question_data["Pred_Rationale"]
                data_dic['Pred_FinalAnswer'] = pred_final_answer
                data_dic['True_FinalAnswer'] = true_final_answer
                dataloader.append(data_dic)


    # questions = [example['question'] for example in dataloader]
    # rationales = [example['rationale'] for example in dataloader]

    total_correct_examples = [example for example in dataloader if example['Pred_FinalAnswer'] == example['True_FinalAnswer']]
    total_incorrect_examples = [example for example in dataloader if example['Pred_FinalAnswer'] != example['True_FinalAnswer']]
    # print('Total nr of examples: ', len(dataloader))
    # print(f'Total nr of correct examples: {len(total_correct_examples)}')
    # print(f'Total nr of incorrect examples: {len(total_incorrect_examples)}')

    correct_questions = [example['question'] for example in total_correct_examples]
    correct_rationales = [example['rationale'] for example in total_correct_examples]

    incorrect_questions = [example['question'] for example in total_incorrect_examples]
    in_correct_rationales = [example['rationale'] for example in total_incorrect_examples]


    # filtered_dataloader = []
    # for example in dataloader:
    #     rationale = example['rationale']
    #     nr_reasonings_steps = len(rationale.replace("\n\n", "\n").split("\n"))
    #     if args.dataset == 'aqua':
    #         nr_reasonings_steps -= 1
            
    #     if len(example['question'].strip().split()) <= args.max_token_len and nr_reasonings_steps <= args.max_ra_len and example['Pred_FinalAnswer'] != "":
    #         filtered_dataloader.append(example)
    
    # print(f'Nr of filtered examples: {len(filtered_dataloader)}')
    # not_filtered_examples = [example for example in dataloader if example not in filtered_dataloader]
    # print(f'Nr of not filtered examples: {len(not_filtered_examples)}')



    # filtered_correct_examples = [example for example in filtered_dataloader if example['Pred_FinalAnswer'] == example['True_FinalAnswer']]
    # filtered_incorrect_examples = [example for example in filtered_dataloader if example['Pred_FinalAnswer'] != example['True_FinalAnswer']]

    # correct_questions = [example['question'] for example in filtered_correct_examples]
    # correct_rationales = [example['Pred_Rationale'] for example in filtered_correct_examples]

    # incorrect_questions = [example['question'] for example in filtered_incorrect_examples]
    # incorrect_rationales = [example['rationale'] for example in filtered_incorrect_examples]

    # # Correct examples
    # # plot nr of reasoning steps for correct rationales

    def filter_outliers(rationales, questions, source='correct'):
        nr_reasoning_steps = [len(rationale.strip().replace("\n\n", "\n").split("\n")) - 1 for rationale in rationales]

        plt.hist(nr_reasoning_steps, bins=10)
        plt.xlabel('Nr of reasoning steps')
        plt.ylabel('Frequency')
        plt.savefig(f'../src/visualizations/plots/{args.dataset}_histogram_nr_reasoningsteps_zeroshotcot_{source}.png')
        plt.show()

        # plot histogram of the nr of tokens of questions
        nr_tokens = [len(question.strip().split()) for question in questions]
        plt.hist(nr_tokens, bins=10)
        plt.xlabel('Nr of tokens in question')
        plt.ylabel('Frequency')
        plt.savefig(f'../src/visualizations/plots/{args.dataset}_question_nrtokens_zeroshotcot_{source}.png')
        plt.show()

    filter_outliers(in_correct_rationales, incorrect_questions, source='incorrect')


    # correct_examples_idx = [example['question_idx'] for example in correct_examples]
    # not_correct_examples_idx = [example['question_idx'] for example in not_correct_examples]
    # random_demo1 = [718, 125, 25, 829] # low acc (4 are correct, 0 incorrect)
    # random_demo2 = [313, 280, 258, 158] # high acc (2 are correct, 2 incorrect)
    # auto_demo1 = [555, 807, 160, 21] # higer acc (3 are correct, 1 incorrect)
    # auto_demo2 = [165, 477, 669, 21] # lower acc (4 correct, 0 incorrect)
    # random_demo3 = [] #  (0 correct, 4 incorrect)
    # random_demo4 = [] #  (0 correct, 4 incorrect)
    # c = 0
    # for idx in auto_demo2:
    #     if idx in not_correct_examples_idx:
    #         c += 1
    # print(c)


    # random.seed(args.random_seed)
    # correct_selected_examples = random.sample(correct_examples, 4)
    # correct_demos = [example for example in correct_selected_examples]
    # not_correct_selected_examples = random.sample(not_correct_examples, 4)
    # not_correct_demos = [example for example in not_correct_selected_examples]
    # correct_demos_dic = {"demo": correct_demos}
    # not_correct_demos_dic = {"demo": not_correct_demos}
    # with open(args.demos_save_dir + '/correct_demos', 'w', encoding="utf-8") as write_f:
    #     json.dump(correct_demos_dic, write_f, indent=4, ensure_ascii=False)
    # with open(args.demos_save_dir + '/not_correct_demos', 'w', encoding="utf-8") as write_f:
    #     json.dump(not_correct_demos_dic, write_f, indent=4, ensure_ascii=False)


    # filtered_examples = [example for example in data_list if example['question_idx'] in [example['question_idx'] for example in filtered_dataloader]]
    
    # c = 0
    # for example in filtered_examples:
    #     if example['Pred_FinalAnswer'] == example['True_FinalAnswer']:
    #         c += 1
    # print(f'Nr of examples from filtered dataloader that have Pred_FinalAnswer = True_FinalAnswer: {c}')


    # not_filtered_examples = [example for example in data_list if example['question_idx'] in [example['question_idx'] for example in not_filtered_examples]]
    # c = 0
    # for example in not_filtered_examples:
    #     if example['Pred_FinalAnswer'] == example['True_FinalAnswer']:
    #         c += 1

    # print(f'Nr of examples from not filtered dataloader that have Pred_FinalAnswer = True_FinalAnswer: {c}')



    # question_idxs = []

    # with open(args.prompt_path, encoding="utf-8") as f:
    #     json_data = json.load(f)
    #     json_data = json_data["demo"]
    #     for line in json_data:
    #         question_idxs.append(line["question_idx"])

    # not_filtered_examples_idxs = [example["question_idx"] for example in not_filtered_examples]
    
    # c = 0
    # for idx in question_idxs:
    #     if idx in not_filtered_examples_idxs:
    #         c +=1

    # print(f'Nr of not filtered examples in the prompt: {c}')

if __name__ == "__main__":
    main()