import argparse
import json 
import random
import os

def parse_arguments():
    parser = argparse.ArgumentParser(description="Random-CoT")
    parser.add_argument(
        "--dataset", type=str, default="aqua",
        choices=["aqua", "gsm8k", "commonsensqa", "addsub", "multiarith", "strategyqa", "svamp", "singleeq", "coin_flip", "last_letters"], help="dataset used for experiment"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="seed for selecting random samples"
    )

    parser.add_argument(
        "--nr_seeds", type=int, default=2, help="maximum number of reasoning chains"
    )

    parser.add_argument(
        "--demos_save_dir", type=str, default="demos/", help="maximum number of reasoning chains"
    )

    parser.add_argument(
        "--dataset_size_limit", type=int, default=100, help="maximum number of reasoning chains"
    )

    parser.add_argument(
        "--nr_demonstrations", type=int, default=3, help="maximum number of reasoning chains"
    )

    args = parser.parse_args()
    if args.dataset == "gsm8k":
        args.training_data_path = "../datasets/gsm8k/train.jsonl"
    elif args.dataset == "aqua":
        args.training_data_path = "../datasets/AQuA/train.json" 

    return args

def main():
    args = parse_arguments()

    if not os.path.exists(args.demos_save_dir):
        os.makedirs(args.demos_save_dir)
        os.makedirs(args.demos_save_dir + 'random')
        os.makedirs(args.demos_save_dir + 'random/' + args.dataset)
    elif not os.path.exists(args.demos_save_dir + 'random'):
        os.makedirs(args.demos_save_dir + 'random')
        os.makedirs(args.demos_save_dir + 'random/' + args.dataset)
    elif not os.path.exists(args.demos_save_dir + 'random/' + args.dataset):
        os.makedirs(args.demos_save_dir + 'random/' + args.dataset)

    args.demos_save_dir = f"{args.demos_save_dir}/random/{args.dataset}/"

    random.seed(args.seed)
    with open(args.training_data_path) as fh:
        train_data = [json.loads(line) for line in fh.readlines() if line]

    if args.dataset_size_limit <= 0:
        args.dataset_size_limit = len(train_data)
    train_data = random.sample(train_data, args.dataset_size_limit)

    for i in range(args.nr_seeds):
        selected_examples = random.sample(train_data, args.nr_demonstrations)
        demos = []
        if args.dataset == 'gsm8k':
            for example in selected_examples:
                question = f"Q: {example['question'].strip()}\nA:"
                rationale = f"Let's think step by step.\n'{example['answer'].split('####')[0].strip()}"
                final_answer = example["answer"].split("#### ")[-1].replace(",", "")
                demo_element = {
                            "question": question,
                            "rationale": rationale,
                            "final_answer": final_answer               
                            }
                demos.append(demo_element)

            demos_dic = {"demo": demos}
            with open(args.demos_save_dir + 'demos' + str(i+1), 'w', encoding="utf-8") as write_f:
                json.dump(demos_dic, write_f, indent=4, ensure_ascii=False)

        elif args.dataset == "aqua":
            for example in selected_examples:
                choices_str =  ' '.join([option for option in example['options']])
                question = f"Q: {example['question'].strip()} Answer Choices: {choices_str}\nA:"
                rationale = f"Let's think step by step.\n{example['rationale']}"
                final_answer = example['correct']
                demo_element = {
                            "question": question,
                            "rationale": rationale,
                            "final_answer": final_answer               
                            }
                demos.append(demo_element)

            demos_dic = {"demo": demos}
            with open(args.demos_save_dir + 'demos' + str(i+1), 'w', encoding="utf-8") as write_f:
                json.dump(demos_dic, write_f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    main()