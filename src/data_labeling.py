import argparse
from utils import *
from generate_demo_active import predict_llm, create_dataloader, answer_extraction, create_several_input_prompts

def parse_arguments():
    parser = argparse.ArgumentParser(description="LLM-Data-Labeling")
    parser.add_argument(
        "--dataset", type=str, default="gsm8k",
        choices=["aqua", "gsm8k"], help="dataset to label"
    )

    #parser.add_argument("--max_num_worker", type=int, default=0, help="maximum number of workers for dataloader")
    parser.add_argument(
        "--model", type=str, default="gpt-4", choices=['gpt-3.5-turbo', 'gpt-4'], help="model used for generating the answers. It is not recommened to use the same llm which was used to select the few-shot demonstrations."
    )

    parser.add_argument(
        "--dir_prompts", type=str, default="prompts_active", help="directory where the prompt file is saved"
    )

    parser.add_argument(
        "--method", type=str, default="zero_shot_cot", choices=["zero_shot_cot", "few_shot_cot",], help="method"
    )
    parser.add_argument(
        "--cot_trigger_no", type=int, default=1, help="A trigger sentence that elicits a model to execute chain of thought"
    )
    # parser.add_argument(
    #     "--max_length_cot", type=int, default=256, help="maximum length of output tokens by model for reasoning extraction"
    # )
    # parser.add_argument(
    #     "--max_length_direct", type=int, default=32, help="maximum length of output tokens by model for answer extraction"
    # )
    parser.add_argument(
        "--limit_dataset_size", type=int, default=0, help="whether to limit the dataset size. if 0, the dataset size is unlimited and we use all the samples in the dataset for labeling."
    )
    # parser.add_argument(
    #     "--api_time_interval", type=float, default=1.0, help=""
    # )
    parser.add_argument(
        "--temperature", type=float, default=0.2, help="temperature used for llm decoding"
    )
    parser.add_argument(
        "--unlabeled_demo_filepath", type=str, default="unlabeled_demos/auto_cot/gsm8k/demos", help="the directory where the unlabeled demos are saved"
    )

    parser.add_argument(
        "--llm_labeled_demos_save_dir", type=str, default="llm_labeled_demos/", help="the directory where the labeled demos should be saved"
    )

    parser.add_argument(
        "--random_seed", type=int, default=42, help="seed for selecting random samples"
    )

    args = parser.parse_args()

    args.direct_answer_trigger_for_fewshot = "The answer is"
    args.direct_answer_trigger_for_zeroshot = "The answer is"
    args.direct_answer_trigger_for_zeroshot_cot = "The answer is"
    args.cot_trigger = "Let's think step by step."
    args.answers_are_available = False
    return args

def main():
    args = parse_arguments()
    if not os.path.exists(args.llm_labeled_demos_save_dir):
        os.makedirs(args.llm_labeled_demos_save_dir)
        os.makedirs(args.llm_labeled_demos_save_dir + args.dataset)
    elif not os.path.exists(args.llm_labeled_demos_save_dir + args.dataset):
        os.makedirs(args.llm_labeled_demos_save_dir + args.dataset)

    args.llm_labeled_demos_save_dir = f"{args.llm_labeled_demos_save_dir}{args.dataset}/"            

    if args.method == "few_shot_cot":
        given_prompt_list = create_several_input_prompts(args, cot_flag=True)
        assert len(given_prompt_list) == 1
        given_prompt = given_prompt_list[0]

    labeled_demos = []
    with open(args.unlabeled_demo_filepath, encoding="utf-8") as f:
        json_data = json.load(f)
        json_data = json_data["demo"]
        for example in json_data:
            if args.method == "few_shot_cot":
                prompt = given_prompt + "Q: " + "{question}" + "\nA: Let's think step by step."
            elif args.method == "zero_shot_cot":
                prompt = "Q: " + "{question}" + "\nA: Let's think step by step."

            response, _, _ = predict_llm(template=prompt, question=example['question'], model=args.model,
                                        temperature=args.temperature)

            # extract the pred answer
            final_answer = answer_extraction(args, response)
            print(f'Question:\n{example["question"]}\n\n')
            print(f'Response:\n{response}\n\n')
            print(f'FA: {final_answer}\n\n')
            print('*' * 60)
            demo_element = {
                            "question_idx": example['question_idx'],
                            "question": example['question'],
                            "rationale": response,
                            "final_answer": final_answer               
                            }
            labeled_demos.append(demo_element)

    labeled_demos = {"demo": labeled_demos}
    with open(args.llm_labeled_demos_save_dir + 'demos', 'w', encoding="utf-8") as write_f:
        json.dump(labeled_demos, write_f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    main()