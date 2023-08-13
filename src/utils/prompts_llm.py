from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import os 
import openai 
from langchain.llms import HuggingFacePipeline, AzureOpenAI 
from langchain.chat_models import AzureChatOpenAI
from constant_vars import *
import json

def create_prompts_inference(args):
    if args.method == 'zero_shot_cot':
        if args.dataset == 'aqua':
            args.prefix = args.prefix + ' If none of options is correct, please choose the option "None of the above".'
        prompts_list = [args.prefix + "\nQ: " + "{question}" + "\nA: Let's think step by step."]
    elif args.method == 'cot':
        args.prefix = args.prefix + ' To generate the answer follow the format of the examples below:\n'        
        prompts_list = create_several_input_prompts(args, cot_flag=True)
    elif args.method == 'standard':
        args.prefix = args.prefix + '\n'
        prompts_list = create_several_input_prompts(args, cot_flag=False)

    return prompts_list


def create_single_input_prompt(args: object, prompt_filename: str, cot_flag:bool)->str:
    """
        Create a prompt for the given dataset
        Args:
            args (object): the arguments passed in from the command line
            prompt_filename (str): the name of the prompt file
            cot_flag (bool): whether to use chain of thought or not
        Returns:
            full prompt (str)
    """
    x, z, y = [], [], []
    
    prompt_path = os.path.join(args.dir_prompts, prompt_filename)
    with open(prompt_path, encoding="utf-8") as f:
        json_data = json.load(f)
        json_data = json_data["demo"]
        for line in json_data:
            x.append(line["question"])
            z.append(line["rationale"])
            y.append(line["final_answer"]) # use final_answer instead of pred_ans

    index_list = list(range(len(x)))
    
    prompt_text = ""
    for i in index_list:
        if cot_flag:
            prompt_text += x[i] + "\n" + z[i] + " " + \
                        args.direct_answer_trigger_for_fewshot + " " + y[i] + ".\n\n"
        else:
            prompt_text += x[i] + "\n" + "A: " + args.direct_answer_trigger_for_fewshot + " " + y[i] + ".\n\n"
    return args.prefix + prompt_text + "Q: " + "{question}" + "\nA: Let's think step by step."

def create_several_input_prompts(args: object, cot_flag:bool) -> list:
    """
        Create a list of prompts for the given dataset
        Args:
            args (object): the arguments passed in from the command line
        Returns:
            prompt_demos (list): the list of prompts
    """
    prompt_demos = []
    for prompt_filename in os.listdir(args.dir_prompts):
        prompt = create_single_input_prompt(args, prompt_filename, cot_flag=cot_flag)
        prompt_demos.append(prompt)
    return prompt_demos


def initialize_llmchain(args) -> LLMChain:
    """
        Run a LLMChain for given a prompt template and question. Return the completion and
        total nr of processed tokens during the run.
        
        Args: 
            args: the arguments passed in from the command line
        Returns:
            llm_chain (LLMChain): the LLMChain object
    """
    prompt = PromptTemplate(input_variables=["question"], template=args.prompt_template)

    if args.model_id.startswith("gpt-35") or args.model_id.startswith('text-davinci') or args.model_id.startswith("gpt-4"):        
        openai.api_key = os.getenv("OPENAI_API_KEY")
        headers = {
            "x-api-key": openai.api_key,
        }

        llm = AzureOpenAI(
        deployment_name=args.model_id,
        model_name=args.model_id,
        temperature=args.temperature,
        headers=headers,
        max_tokens=1024,
        )
    else:
        llm = HuggingFacePipeline.from_model_id(
        model_id=args.model_id,
        model_kwargs={"temperature": args.temperature,
                     "trust_remote_code": True,
                     "max_seq_len": 4096}, # max_length
        )
    
    args.llm_chain = LLMChain(prompt=prompt, llm=llm, verbose=False)

def build_prefix(args):
    """
        Build the prefix for the given dataset
        Args:
            args (object): the arguments passed in from the command line
    """
    if args.dataset == "gsm8k":
        args.prefix = prefix_gsm8k
    elif args.dataset == "aqua":
        args.prefix = prefix_aqua
    else:
        raise NotImplementedError("dataset not implemented")

def build_prompt_template(args: object):
    """
        Build the prompt template for the given dataset and method
        Args:
            args (object): the arguments passed in from the command line
    """
    if args.method == "few_shot_cot":
        args.prefix = args.prefix + ' To generate the answer follow the format of the examples below:\n'
        given_prompt_list = create_several_input_prompts(args, cot_flag=True)
        assert len(given_prompt_list) == 1
        args.prompt_template = given_prompt_list[0]
    elif args.method == "zero_shot_cot":
        args.prompt_template = args.prefix + "\nQ: " + "{question}" + "\nA: Let's think step by step."
    
    print(f'PROMPT TEMPLATE:\n{args.prompt_template}\n')

def build_prompt_initialize_llmchain(args):
    """
        Build the prefix, prompt template and initialize the LLMChain
        Args:
            args (object): the arguments passed in from the command line
    """
    build_prefix(args)
    build_prompt_template(args)
    initialize_llmchain(args)

