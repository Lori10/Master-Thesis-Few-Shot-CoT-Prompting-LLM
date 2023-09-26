import json
import re 
import sys 

all_records_path = 'inference_results/aqua/auto/2023_09_24_16_49_35/QA_record_prompt1.txt'
wrong_records_path = 'inference_results/aqua/auto/2023_09_24_16_49_35/wrong_prompt1.txt'

with open(all_records_path, 'r', encoding='utf-8') as file:
    all_records_data = file.read()

with open(wrong_records_path, 'r', encoding='utf-8') as file1:
    wrong_records_data = file1.readlines()

# Parse the JSON data
parsed_all_records = json.loads(all_records_data)
list_all_records = parsed_all_records[1:]

pattern = r"'question_idx':\s*(\d+)"

wrong_idxs = []
for line in wrong_records_data:
    match = re.search(pattern, line)

    if match:
        q_id = int(match.group(1))
        wrong_idxs.append(q_id)
    else:
        print(f"No 'question_idx' found in the input string: {line}")

right_idxs = [idx for idx in range(len(list_all_records)) if idx not in wrong_idxs]

for idx in right_idxs:
    print(f'Question IDX: {idx}' + '\n')

    example = list_all_records[idx][0]
    print('Question:')
    print(example['Question'] + '\n')

    print('Predicted Rationale:', example['Pred_Rationale'] + '\n')

    print('Predicted Answer:')
    print(example['Pred_FinalAnswer'] + '\n')

    print('True Answer:')
    print(example['True_FinalAnswer'] + '\n')

    print('*' * 50)

    
