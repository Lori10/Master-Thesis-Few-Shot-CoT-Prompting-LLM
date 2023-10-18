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
        "--dataset", type=str, default="aqua",
        choices=["aqua", "gsm8k"], help="dataset used for experiment"
    )

    parser.add_argument(
        "--data_path", type=str, default="../datasets/gpt35_zeroshotcot_training_data/aqua/QA_record_prompt1.txt",
        choices=["../datasets/original/gsm8k/train.jsonl", "../datasets/original/AQuA/train.json",
                 "../datasets/gpt35_zeroshotcot_training_data/gsm8k/QA_record_prompt1.txt",
                 "../datasets/gpt35_zeroshotcot_training_data/aqua/QA_record_prompt1.txt"], help="dataset used for experiment"
    )

    parser.add_argument("--random_seed", type=int, default=1, help="random seed")

    parser.add_argument(
        "--dataset_size_limit", type=int, default=1000, help="whether to limit training dataset size. if 0, the dataset size is unlimited and we use all the samples in the dataset for creating the demonstrations."
    )

    parser.add_argument(
        "--max_token_len", type=float, default=70, help="maximum number of reasoning chains"
    )

    parser.add_argument(
        "--max_ra_len", type=float, default=15, help="maximum number of reasoning chains"
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

    
    
    # if args.data_path.endswith('json') or args.data_path.endswith('jsonl'):
    #     dataloader = create_dataloader(args)
    # else:

    #     with open(args.data_path, 'r') as f:
    #         data = json.load(f)

    #     dataloader = []
    #     for entry in data[1:]:
    #         for question_data in entry:
    #             question_idx = question_data["Question_idx"]
    #             pred_final_answer = question_data["Pred_FinalAnswer"]
    #             true_final_answer = question_data["True_FinalAnswer"]

    #             data_dic = {}
    #             data_dic['question_idx'] = question_idx
    #             data_dic['question'] = question_data['Question']
    #             data_dic["rationale"] = question_data["Pred_Rationale"]
    #             data_dic['Pred_FinalAnswer'] = pred_final_answer
    #             data_dic['True_FinalAnswer'] = true_final_answer
    #             dataloader.append(data_dic)

    def read_data(zeroshotcot_data_path, true_data_path):
        
        args.data_path = true_data_path
        true_dataloader = create_dataloader(args)

        with open(zeroshotcot_data_path, 'r') as f:
            data = json.load(f)

        zeroshotcot_dataloader = []
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
                zeroshotcot_dataloader.append(data_dic)

        return true_dataloader, zeroshotcot_dataloader

    aqua_zeroshotcot_data_path = "../datasets/gpt35_zeroshotcot_training_data/aqua/QA_record_prompt1.txt"
    aqua_true_data_path = "../datasets/original/AQuA/train.json"
    true_dataloader, zeroshotcot_dataloader = read_data(aqua_zeroshotcot_data_path, aqua_true_data_path)

    # gsm8k_true_data_path = "../datasets/original/gsm8k/train.jsonl"
    # gsm8k_zeroshotcot_data_path = "../datasets/gpt35_zeroshotcot_training_data/gsm8k/QA_record_prompt1.txt"
    # true_dataloader, zeroshotcot_dataloader = read_data(gsm8k_zeroshotcot_data_path, gsm8k_true_data_path)

    # true_rationales = [example['rationale'] for example in true_dataloader]
    # zeroshotcot_rationales = [example['rationale'] for example in zeroshotcot_dataloader]

    # DISTRIBUTON OF NR_TOKENS_QUESTION FOR TRUE AND ZEROSHOTCOT DATA
    # true_questions = [example['question'] for example in true_dataloader]
    # true_questions = [len(question.strip().split()) for question in true_questions]
    # zeroshotcot_questions = [example['question'] for example in zeroshotcot_dataloader]
    # zeroshotcot_questions = [len(question.strip().split()) for question in zeroshotcot_questions]
    # def compute_quantiles(data):
    #     five_main_quantiles = np.percentile(data, [0, 25, 50, 75, 100])
    #     iqr = five_main_quantiles[3] - five_main_quantiles[1]
    #     lower_limit = five_main_quantiles[1] - 1.5 * iqr
    #     upper_limit = five_main_quantiles[3] + 1.5 * iqr
    #     return five_main_quantiles, iqr, lower_limit, upper_limit
    # # quantiles_nr_tokens, iqr_nr_tokens, lower_limit_nr_tokens, upper_limit_nr_tokens = compute_quantiles(true_questions)
    # # quantiles_dic = {
    # #         "quantiles": quantiles_nr_tokens.tolist(),
    # #         "iqr": iqr_nr_tokens,
    # #         "lower_limit": lower_limit_nr_tokens,
    # #         "upper_limit": upper_limit_nr_tokens
    # #     }
    # # with open(f'../src/visualizations/plots/{args.dataset}_quantiles.json', 'w') as outfile:
    # #     json.dump(quantiles_dic, outfile, indent=4)
    # sns.boxplot(true_questions)
    # #sns.histplot(true_questions, bins=10)
    # plt.xlabel('Nr of tokens in question')
    # plt.ylabel('Frequency')
    # plt.savefig('../src/visualizations/plots/fig_1.png')
    # plt.show()


    # total_correct_examples = [example for example in zeroshotcot_dataloader if example['Pred_FinalAnswer'] == example['True_FinalAnswer']]
    # total_incorrect_examples = [example for example in zeroshotcot_dataloader if example['Pred_FinalAnswer'] != example['True_FinalAnswer']]
    # print('Total nr of examples: ', len(zeroshotcot_dataloader))
    # print(f'Total nr of correct examples: {len(total_correct_examples)}')
    # print(f'Total nr of incorrect examples: {len(total_incorrect_examples)}')
    # correct_questions = [example['question'] for example in total_correct_examples]
    # correct_rationales = [example['rationale'] for example in total_correct_examples]
    # incorrect_questions = [example['question'] for example in total_incorrect_examples]
    # incorrect_rationales = [example['rationale'] for example in total_incorrect_examples]

    filtered_dataloader = []
    for example in zeroshotcot_dataloader:
        rationale = example['rationale']
        nr_reasonings_steps = len(rationale.replace("\n\n", "\n").split("\n"))
        if args.dataset == 'aqua':
            nr_reasonings_steps -= 1
        
        # if example['Pred_FinalAnswer'] != "":
        #     filtered_dataloader.append(example)

        if len(example['question'].strip().split()) <= args.max_token_len and nr_reasonings_steps <= args.max_ra_len and example['Pred_FinalAnswer'] != "":
            filtered_dataloader.append(example)

    print('Filter details:')
    print('Max token len: ', args.max_token_len)
    print('Max reasoning steps: ', args.max_ra_len)
    print('------------------------')
    print(f'Nr of filtered examples: {len(filtered_dataloader)}')
    not_filtered_examples = [example for example in zeroshotcot_dataloader if example not in filtered_dataloader]
    print(f'Nr of not filtered examples: {len(not_filtered_examples)}')


    filtered_correct_examples = [example for example in filtered_dataloader if example['Pred_FinalAnswer'] == example['True_FinalAnswer']]
    filtered_incorrect_examples = [example for example in filtered_dataloader if example['Pred_FinalAnswer'] != example['True_FinalAnswer']]
    print('Filtered nr of correct examples: ', len(filtered_correct_examples))
    print('Filtered % of correct examples: ', len(filtered_correct_examples)/len(filtered_dataloader))
    print('Filtered nr of incorrect examples: ', len(filtered_incorrect_examples))
    print('Filtered % of incorrect examples: ', len(filtered_incorrect_examples)/len(filtered_dataloader))


    # correct_questions = [example['question'] for example in filtered_correct_examples]
    # correct_rationales = [example['Pred_Rationale'] for example in filtered_correct_examples]

    # incorrect_questions = [example['question'] for example in filtered_incorrect_examples]
    # incorrect_rationales = [example['rationale'] for example in filtered_incorrect_examples]

    # # Correct examples
    # # plot nr of reasoning steps for correct rationales

    # def filter_outliers(rationales, questions, source='correct'):
    #     nr_reasoning_steps = [len(rationale.strip().replace("\n\n", "\n").split("\n")) - 1 for rationale in rationales]

    #     quantiles_nr_reasoning_steps = np.percentile(nr_reasoning_steps, [0, 25, 50, 70, 75, 100])
    #     iqr_nr_reasoning_steps = quantiles_nr_reasoning_steps[4] - quantiles_nr_reasoning_steps[1]
    #     lower_limit = quantiles_nr_reasoning_steps[1] - 1.5 * iqr_nr_reasoning_steps
    #     upper_limit = quantiles_nr_reasoning_steps[4] + 1.5 * iqr_nr_reasoning_steps

    #     print('IQR nr of reasoning steps: ', iqr_nr_reasoning_steps)
    #     print('Nr of reasoning steps Quantiles:')
    #     print(quantiles_nr_reasoning_steps)
    #     print('Nr of reasoning steps lower limit:')
    #     print(lower_limit)
    #     print('Nr of reasoning steps upper limit:')
    #     print(upper_limit)

    #     plt.hist(nr_reasoning_steps, bins=10)
    #     #sns.boxplot(nr_reasoning_steps)
    #     plt.xlabel('Nr of reasoning steps')
    #     plt.ylabel('Frequency')
    #     plt.title('AQUA/ALL')
    #     plt.savefig(f'../src/visualizations/plots/{args.dataset}_histogram_nr_reasoningsteps_zeroshotcot_{source}.png')
    #     plt.show()

    #     # plot histogram of the nr of tokens of questions
    #     nr_tokens = [len(question.strip().split()) for question in questions]
    #     quantiles_nrtokens = np.percentile(nr_tokens, [0, 25, 50, 70, 75, 100])
    #     iqr_nrtokens = quantiles_nrtokens[4] - quantiles_nrtokens[1]
    #     lower_limit = quantiles_nrtokens[1] - 1.5 * iqr_nrtokens
    #     upper_limit = quantiles_nrtokens[4] + 1.5 * iqr_nrtokens
    #     print('Nr of Tokens Question Quantiles:')
    #     print(quantiles_nrtokens)
    #     print('Nr of reasoning steps lower limit:')
    #     print(lower_limit)
    #     print('Nr of reasoning steps upper limit:')
    #     print(upper_limit)

    #     plt.hist(nr_tokens, bins=10)
    #     #sns.boxplot(nr_tokens)
    #     plt.xlabel('Nr of tokens in question')
    #     plt.ylabel('Frequency')
    #     plt.title('AQUA/ALL')
    #     plt.savefig(f'../src/visualizations/plots/{args.dataset}_question_nrtokens_zeroshotcot_{source}.png')
    #     plt.show()



    def correct_vs_incorrect_analysis(rationale_1, question_1, rationale_2, question_2):
        def get_nr_reasoning_steps(rationales):
            return [len(rationale.strip().replace("\n\n", "\n").split("\n")) - 1 for rationale in rationales]

        def get_nrtokens_questions(questions):
            return [len(question.strip().split()) for question in questions]


        def compute_quantiles(data):
            five_main_quantiles = np.percentile(data, [0, 25, 50, 75, 100])
            iqr = five_main_quantiles[3] - five_main_quantiles[1]
            lower_limit = five_main_quantiles[1] - 1.5 * iqr
            upper_limit = five_main_quantiles[3] + 1.5 * iqr
            return five_main_quantiles, iqr, lower_limit, upper_limit
        
        nr_reasoning_steps_1 = get_nr_reasoning_steps(rationale_1)
        nr_tokens_1 = get_nrtokens_questions(question_1)
        nr_reasoning_steps_2 = get_nr_reasoning_steps(rationale_2)
        nr_tokens_2 = get_nrtokens_questions(question_2)

        quantiles_nr_reasoning_steps_1, iqr_nr_reasoning_steps_1, lower_limit_nr_reasoning_steps_1, upper_limit_nr_reasoning_steps_1 = compute_quantiles(nr_reasoning_steps_1)
        quantiles_nr_tokens_1, iqr_nr_tokens_1, lower_limit_nr_tokens_1, upper_limit_nr_tokens_1 = compute_quantiles(nr_tokens_1)

        quantiles_nr_reasoning_steps_2, iqr_nr_reasoning_steps_2, lower_limit_nr_reasoning_steps_2, upper_limit_nr_reasoning_steps_2 = compute_quantiles(nr_reasoning_steps_2)
        quantiles_nr_tokens_2, iqr_nr_tokens_2, lower_limit_nr_tokens_2, upper_limit_nr_tokens_2 = compute_quantiles(nr_tokens_2)

        nr_reasoning_steps_1_dict = {
            "quantiles": quantiles_nr_reasoning_steps_1.tolist(),
            "iqr": iqr_nr_reasoning_steps_1,
            "lower_limit": lower_limit_nr_reasoning_steps_1,
            "upper_limit": upper_limit_nr_reasoning_steps_1
        }

        nr_tokens_1_dict = {
            "quantiles": quantiles_nr_tokens_1.tolist(),
            "iqr": iqr_nr_tokens_1,
            "lower_limit": lower_limit_nr_tokens_1,
            "upper_limit": upper_limit_nr_tokens_1
        }

        nr_reasoning_steps_2_dict = {
            "quantiles": quantiles_nr_reasoning_steps_2.tolist(),
            "iqr": iqr_nr_reasoning_steps_2,
            "lower_limit": lower_limit_nr_reasoning_steps_2,
            "upper_limit": upper_limit_nr_reasoning_steps_2
        }

        nr_tokens_2_dict = {
            "quantiles": quantiles_nr_tokens_2.tolist(),
            "iqr": iqr_nr_tokens_2,
            "lower_limit": lower_limit_nr_tokens_2,
            "upper_limit": upper_limit_nr_tokens_2
        }


        # Export the values to a JSON file
        with open(f'../src/visualizations/plots/{args.dataset}_nrtokenquestion_zeroshotcot_correct.json', 'w') as outfile:
            json.dump(nr_tokens_1_dict, outfile, indent=4)

        with open(f'../src/visualizations/plots/{args.dataset}_nrreasoningsteps_zeroshotcot_correct.json', 'w') as outfile:
            json.dump(nr_reasoning_steps_1_dict, outfile, indent=4)

        with open(f'../src/visualizations/plots/{args.dataset}_nrtokenquestion_zeroshotcot_incorrect.json', 'w') as outfile:
            json.dump(nr_tokens_2_dict, outfile,  indent=4)

        with open(f'../src/visualizations/plots/{args.dataset}_nrreasoningsteps_zeroshotcot_incorrect.json', 'w') as outfile:
            json.dump(nr_reasoning_steps_2_dict, outfile, indent=4)
        

        # plt.figure(figsize=(10, 6))  # Adjust the figure size as needed
        # # First subplot
        # plt.subplot(2, 2, 1)
        # sns.boxplot(nr_reasoning_steps_1)
        # #sns.histplot(nr_reasoning_steps_1, bins=10)
        # plt.xlabel('Nr of reasoning steps')
        # plt.ylabel('Frequency')
        # plt.title('Correctly Classified')

        # # Second subplot
        # plt.subplot(2, 2, 2)
        # sns.boxplot(nr_reasoning_steps_2)
        # #sns.histplot(nr_reasoning_steps_2, bins=10)
        # plt.xlabel('Nr of reasoning steps')
        # plt.ylabel('Frequency')
        # plt.title('Incorrectly Classified')

        # # Third subplot
        # plt.subplot(2, 2, 3)
        # sns.boxplot(nr_tokens_1)
        # #sns.histplot(nr_tokens_1, bins=10)
        # plt.xlabel('Nr of tokens in question')
        # plt.ylabel('Frequency')
        # plt.title('Correctly Classified')

        # # Fourth subplot
        # plt.subplot(2, 2, 4)
        # sns.boxplot(nr_tokens_2)
        # #sns.histplot(nr_tokens_2, bins=10)
        # plt.xlabel('Nr of tokens in question')
        # plt.ylabel('Frequency')
        # plt.title('Incorrectly Classified')

        # plt.tight_layout()  # Adjust the layout to prevent overlapping
        # plt.savefig(f'../src/visualizations/plots/{args.dataset}_distribution_questionnrtokens_nrreasoningsteps_correct_vs_incorrect_.png', dpi=300, bbox_inches='tight')
        # plt.show()
    
    #correct_vs_incorrect_analysis(correct_rationales, correct_questions, incorrect_rationales, incorrect_questions)




    def true_vs_zeroshotcot_nr_intermediate_steps(rationale_1, rationale_2):
        def get_nr_reasoning_steps(rationales):
            return [len(rationale.strip().replace("\n\n", "\n").split("\n")) - 1 for rationale in rationales]

        def compute_quantiles(data):
            five_main_quantiles = np.percentile(data, [0, 25, 50, 75, 100])
            iqr = five_main_quantiles[3] - five_main_quantiles[1]
            lower_limit = five_main_quantiles[1] - 1.5 * iqr
            upper_limit = five_main_quantiles[3] + 1.5 * iqr
            return five_main_quantiles, iqr, lower_limit, upper_limit
        
        nr_reasoning_steps_1 = get_nr_reasoning_steps(rationale_1)
        nr_reasoning_steps_2 = get_nr_reasoning_steps(rationale_2)

        quantiles_nr_reasoning_steps_1, iqr_nr_reasoning_steps_1, lower_limit_nr_reasoning_steps_1, upper_limit_nr_reasoning_steps_1 = compute_quantiles(nr_reasoning_steps_1)
        quantiles_nr_reasoning_steps_2, iqr_nr_reasoning_steps_2, lower_limit_nr_reasoning_steps_2, upper_limit_nr_reasoning_steps_2 = compute_quantiles(nr_reasoning_steps_2)

        nr_reasoning_steps_1_dict = {
            "quantiles": quantiles_nr_reasoning_steps_1.tolist(),
            "iqr": iqr_nr_reasoning_steps_1,
            "lower_limit": lower_limit_nr_reasoning_steps_1,
            "upper_limit": upper_limit_nr_reasoning_steps_1
        }

        nr_reasoning_steps_2_dict = {
            "quantiles": quantiles_nr_reasoning_steps_2.tolist(),
            "iqr": iqr_nr_reasoning_steps_2,
            "lower_limit": lower_limit_nr_reasoning_steps_2,
            "upper_limit": upper_limit_nr_reasoning_steps_2
        }

        with open(f'../src/visualizations/plots/{args.dataset}_nrreasoningsteps_truedata.json', 'w') as outfile:
            json.dump(nr_reasoning_steps_1_dict, outfile, indent=4)


        with open(f'../src/visualizations/plots/{args.dataset}_nrreasoningsteps_zeroshotcot_data.json', 'w') as outfile:
            json.dump(nr_reasoning_steps_2_dict, outfile, indent=4)
        

        # plt.figure(figsize=(10, 4))  # Adjust the figure size as needed
        # # First subplot
        # plt.subplot(121)
        # sns.boxplot(nr_reasoning_steps_1)
        # #sns.histplot(nr_reasoning_steps_1, bins=10)
        # plt.xlabel('Nr of reasoning steps')
        # plt.ylabel('Frequency')
        # plt.title('True Labels')

        # # Second subplot
        # plt.subplot(122)
        # sns.boxplot(nr_reasoning_steps_2)
        # #sns.histplot(nr_reasoning_steps_2, bins=10)
        # plt.xlabel('Nr of reasoning steps')
        # plt.ylabel('Frequency')
        # plt.title('ZeroShotCoT Labels')


        # plt.tight_layout()  # Adjust the layout to prevent overlapping
        # plt.savefig(f'../src/visualizations/plots/{args.dataset}_distribution_questionnrtokens_nrreasoningsteps_correct_vs_incorrect_.png', dpi=300, bbox_inches='tight')
        # plt.show()


    #true_vs_zeroshotcot_nr_intermediate_steps(true_rationales, zeroshotcot_rationales)

















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