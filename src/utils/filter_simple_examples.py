def filter_examples_with_labels(dataloader, max_token_len=60, max_ra_len=5):
    filtered_examples = []
    for example in dataloader:
        if len(example['question'].strip().split()) <= max_token_len and len(example['rationale'].replace("\n\n", "\n").split("\n")) <= max_ra_len and example['final_answer'] != "":
            rationale = rationale.replace("\n\n", "\n").replace("\n", " ").strip()
            rationale = " ".join(rationale.split())

            filtered_examples.append({
                "question_idx" : example['question_idx'],
                "question": example['question'],
                "rationale": example['rationale'],
                "final_answer": example['final_answer'],
                }
            )
    return filtered_examples

def filter_examples_no_labels(dataloader, max_token_len=60):
    filtered_examples = []
    for example in dataloader:
        if len(example['question'].strip().split()) <= max_token_len:        
            filtered_examples.append(example)
    return filtered_examples