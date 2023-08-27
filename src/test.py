import load_env_vars
import argparse
from utils.prompts_llm import create_prompts_inference, initialize_llm
from utils.inference_llm import initialize_llmchain
from utils.final_answer_extraction import run_llm_extract_answer

def parse_arguments():
    parser = argparse.ArgumentParser(description="Uncertainty-Estimator")
    parser.add_argument(
        "--dataset", type=str, default="gsm8k",
        choices=["aqua", "gsm8k", "commonsensqa", "addsub", "multiarith", "strategyqa", "svamp", "singleeq", "coin_flip", "last_letters"], help="dataset used for experiment"
    )

    parser.add_argument(
        "--data_path", type=str, default="../datasets/gsm8k/train.jsonl",
        choices=["../datasets/gsm8k/train.jsonl", "../datasets/AQuA/train.json"], help="dataset used for experiment"
    )

    parser.add_argument(
        "--random_seed", type=int, default=1, help="seed for selecting random samples"
    )

    parser.add_argument(
        "--dataset_size_limit", type=int, default=3, help="whether to limit training dataset size. if 0, the dataset size is unlimited and we use all the samples in the dataset for creating the demonstrations."
    )

    parser.add_argument(
        "--sort_by", type=str, default='entropy', choices=['disagreement', 'variance', 'entropy'], help="sort the final result by given option"
    )

    parser.add_argument(
        "--num_trails", type=int, default=5, help="number of trails to run for each qeestion"
    )

    parser.add_argument(
        "--temperature", type=float, default=0.7, help="temperature for llm decoding"
    )

    parser.add_argument(
        "--model_id", type=str, default="gpt-35-turbo-0613", choices=["gpt-35-turbo-0613" ,"text-davinci-003", "tiiuae/falcon-7b-instruct"], help="model used for decoding."
    )

    parser.add_argument(
        "--method", type=str, default="cot", choices=["zero_shot_cot", "standard", "cot"], help="method"
    )

    parser.add_argument(
        "--dir_prompts", type=str, default="uncertainty_estimation_prompts/gsm8k", help="prompts to use"
    )

    parser.add_argument(
        "--answers_are_available", type=bool, default=True, help='true if answers are available in the test dataset, false otherwise'
    )

    parser.add_argument(
        "--uncertainty_save_dir", type=str, default="uncertainties", help="output directory"
    )

    args = parser.parse_args()

    if args.dataset == "gsm8k":
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif args.dataset == "aqua":
        args.direct_answer_trigger = "\nThe answer is"
    else:
        raise ValueError("dataset is not properly defined ...")

    trigger = args.direct_answer_trigger.replace("\nTherefore, ", "")
    args.direct_answer_trigger_for_zeroshot = trigger[0].upper() + trigger[1:]
    args.direct_answer_trigger_for_zeroshot_cot = args.direct_answer_trigger
    args.direct_answer_trigger_for_fewshot = "The answer is"
    args.cot_trigger = "Let's think step by step."

    return args

def main():
    args = parse_arguments()
    question = 'Q: James takes 2 Tylenol tablets that are 375 mg each, every 6 hours.  How many mg does he take a day?'

    prompts_list = create_prompts_inference(args)
    initialize_llmchain(args, prompts_list[0], llm_init=False)
    num_trails = 5
    i = 1

    while i<=num_trails:
        try:
            print(f'Index: {i}')
            response = run_llm_extract_answer(args, question)
            print(response)
            print('-'*50)

        except Exception as e:
            print(f'Error message : {e}')
            i -= 1

        i += 1


    


if __name__ == '__main__':
    main()



