def filter_examples_with_labels(args, dataloader, max_token_len, max_ra_len):
    filtered_examples = []
    for example in dataloader:
        rationale = example['rationale']
        nr_reasonings_steps = len(rationale.replace("\n\n", "\n").split("\n"))
        nr_reasonings_steps -= 1
        if args.dataset == 'aqua':
            nr_reasonings_steps -= 1

        if len(example['question'].strip().split()) <= max_token_len and nr_reasonings_steps <= max_ra_len and example['final_answer'] != "":
            rationale = rationale.replace("\n\n", "\n").replace("\n", " ").strip()
            rationale = " ".join(rationale.split())

            example['rationale'] = rationale
            filtered_examples.append(example)
    return filtered_examples

def filter_examples_no_labels(dataloader, max_token_len=60):
    filtered_examples = []
    for example in dataloader:
        if len(example['question'].strip().split()) <= max_token_len:        
            filtered_examples.append(example)
    return filtered_examples