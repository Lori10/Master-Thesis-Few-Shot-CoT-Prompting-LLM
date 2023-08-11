import re
from collections import Counter

def answer_extraction(args: object, responses: str):
    """
        Extract the answer from the response
        Args:
            args (object): the arguments passed in from the command line
            responses (str): the response from the model
        Returns:
            answer (str): the answer extracted from the response
    """
    pred_ans = ""
    temp = responses

    if args.dataset in ("gsm8k"):
        temp = temp.replace(",", "")
        temp = [s for s in re.findall(r'-?\d+\.?\d*', temp)]
    elif args.dataset in ("aqua"):
        if 'none of the answer choices are correct' in responses.lower() or 'The answer is not given in the answer choices' in responses.lower() or 'The answer is not given in the answer choices' in responses.lower() or "The answer is not listed in the answer choices" in responses.lower():
            return 'No answer'
        temp = re.findall(r'A|B|C|D|E', temp)
    
    if len(temp) != 0:
        answer = temp[-1]
        # if there is . at the end of answer, remove it
        # e.g. answer = 64.
        if answer != "":
            if answer[-1] == ".":
                answer = answer[:-1]

        # round the answer to nearest integer
        if args.dataset in ("gsm8k"):
            try:
                answer = str(round(float(answer)))
            except:
                answer = "" # no sol or sol doesn't have valid format
        pred_ans = answer
    else:
        pred_ans = ""

    return pred_ans

def find_most_frequent(arr, n):
    """
        Find the most frequent answer in the given array of answers
    """
    arr_acounts = Counter(arr[:n])
    most_frequent_item, frequency = arr_acounts.most_common(1)[0]
    return frequency, most_frequent_item

def run_llm_extract_answer(args, question):
    """
        Run a LLMChain for given a prompt template and question. Return the final answer and completion
    """
    response = args.llm_chain.run(question=question)
    return answer_extraction(args, response), response