from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import os 
import openai 
from langchain.llms import HuggingFacePipeline, AzureOpenAI 
from langchain.chat_models import AzureChatOpenAI, ChatOpenAI
from constant_vars import *
import json
from langchain.prompts import ChatPromptTemplate
from langchain.prompts.chat import SystemMessage, HumanMessagePromptTemplate, HumanMessage, AIMessage
from env_vars import *
from transformers import StoppingCriteria, StoppingCriteriaList, AutoTokenizer, AutoModelForCausalLM, pipeline
import torch 
from torch import cuda, bfloat16
from transformers import BitsAndBytesConfig
from langchain.llms import VLLM

def from_chatmodelmessages_to_string(messages_prompt):
    return '\n\n'.join([message.content for message in messages_prompt[:-1]]) + '\n\n'  + messages_prompt[-1].prompt.template 

def create_prompts_inference(args):
    build_prefix(args)
    
    args.suffix = "\nQ: " + "{question}" + "\nA: Let's think step by step."
    if args.method == 'zero_shot_cot':  
        if args.model_id.startswith("gpt-35") or args.model_id.startswith("gpt-4") or args.model_id.startswith("gpt-3.5"):
            return [create_prompttemplate_chatmodel(None, args)]
        else:
            return [PromptTemplate(input_variables=["question"], template=args.prefix + args.suffix)]
    else:
        args.prefix = args.prefix + ' To generate the answer follow the format of the examples below:\n'        
        if args.method == 'cot':
            prompts_list = create_several_input_prompts(args, cot_flag=True)
        elif args.method == 'standard':
            prompts_list = create_several_input_prompts(args, cot_flag=False)

        if args.model_id.startswith("gpt-35") or args.model_id.startswith("gpt-4") or args.model_id.startswith("gpt-3.5"):
            prompts_list = [create_prompttemplate_chatmodel(prompt, args) for prompt in prompts_list]
        else:
            prompts_list = [PromptTemplate(input_variables=["question"], template=prompt) for prompt in prompts_list]
    
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
            y.append(line["final_answer"]) 

    index_list = list(range(len(x)))
    
    prompt_text = ""
    for i in index_list:
        if cot_flag:
            prompt_text += x[i] + "\n" + z[i] + " " + \
                        args.direct_answer_trigger_for_fewshot + " " + y[i] + ".\n\n"
        else:
            prompt_text += x[i] + "\n" + "A: " + args.direct_answer_trigger_for_fewshot + " " + y[i] + ".\n\n"
    
    return args.prefix + prompt_text + args.suffix

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

def create_prompt_template_other_models(prompt, _):
    return PromptTemplate(input_variables=["question"], template=prompt)

def create_prompttemplate_chatmodel(prompt: str, args):
    messages = [SystemMessage(content=(args.prefix.strip()))]
    if args.method in ['cot', 'standard']:
        start = prompt.find('Q: ')
        end = prompt.rfind('Q: ')
        few_shot_examples_str = prompt[start:end]
        few_shot_examples_list = [example.strip() for example in few_shot_examples_str.split('\n\n') if example.strip() != '']
        
        for example in few_shot_examples_list:
            start = example.find('Q: ')
            end = example.rfind('A: ')
            example_question = example[start:end].strip()
            example_answer = example[end:].strip()
            messages += [HumanMessage(content=example_question), AIMessage(content=example_answer)]

    return ChatPromptTemplate.from_messages(messages + [HumanMessagePromptTemplate.from_template(args.suffix.strip())]) 

def create_header_llm():
    #openai.api_key = os.getenv("OPENAI_API_KEY")
    headers = {
        "x-api-key": AZURE_OPENAI_API_KEY,
    }
    return headers

def initialize_llm(args, model_id='gpt-3.5-turbo-0613'):
    if model_id == 'tiiuae/falcon-40b-instruct':
        llm = VLLM(model=model_id,
        tensor_parallel_size=4, # number of GPUs available
        trust_remote_code=True,  # mandatory for hf models
        torch_dtype=torch.bfloat16,
        max_new_tokens=300,
        temperature=args.temperature,
        vllm_kwargs={"gpu_memory_utilization":1.0}
        )

    elif model_id == 'tiiuae/falcon-7b-instruct':
        llm = VLLM(model=model_id,
        tensor_parallel_size=1, # number of GPUs available
        trust_remote_code=True,  # mandatory for hf models
        torch_dtype=torch.bfloat16,
        max_new_tokens=300,
        temperature=args.temperature,
        vllm_kwargs={"gpu_memory_utilization":1.0}
        )

    elif model_id.startswith("gpt-35"):
        headers = create_header_llm()

        llm = AzureChatOpenAI(
        deployment_name=model_id,
        model_name=model_id,
        temperature=args.temperature,
        headers=headers,
        max_tokens=1024,
        openai_api_base=OPENAI_API_BASE,
        openai_api_type=OPENAI_API_TYPE,
        openai_api_version=OPENAI_API_VERSION,
        openai_api_key=AZURE_OPENAI_API_KEY
        )
    elif model_id.startswith("gpt-3.5"):
        llm = ChatOpenAI(
                model_name='gpt-3.5-turbo-0613',
                temperature=args.temperature,
                max_tokens=1024,
                openai_api_key=OPENAI_API_KEY
                )

    elif model_id == 'gpt-4':
        llm = ChatOpenAI(
                model_name=model_id,
                temperature=args.temperature,
                max_tokens=1024,
                openai_api_key=OPENAI_API_KEY
            )
    else:
        raise NotImplementedError(f'Model {model_id} not supported')

    return llm

def initialize_llmchain(llm, prompt_template):
    return LLMChain(prompt=prompt_template, llm=llm, verbose=False)

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

