import sys
import json
import numpy as np 

def inference_save_info(args, correct_list, wrong_list, QA_record_list, prompts_list, len_dataloader):
    acc_prompt_list = []
    if args.output_dir is not None:
        for i in range(len(correct_list)):
            if prompts_list:
                acc_prompt_dic = {'prompt' : prompts_list[i],
                                'accuracy': correct_list[i] / len_dataloader}
            else:
                acc_prompt_dic = {'accuracy': correct_list[i] / len_dataloader}

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