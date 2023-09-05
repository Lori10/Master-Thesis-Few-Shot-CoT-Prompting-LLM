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

def from_chatmodelmessages_to_string(messages_prompt):
    return '\n\n'.join([message.content for message in messages_prompt[:-1]]) + '\n\n'  + messages_prompt[-1].prompt.template 

def create_prompts_inference(args):
    build_prefix(args)
    
    args.suffix = "\nQ: " + "{question}" + "\nA: Let's think step by step."
    if args.method == 'zero_shot_cot':  
        if args.model_id.startswith("gpt-35"):
            return [create_prompt_template_gpt35(None, args)]
        else:
            return [PromptTemplate(input_variables=["question"], template=args.prefix + args.suffix)]
    else:
        args.prefix = args.prefix + ' To generate the answer follow the format of the examples below:\n'        
        if args.method == 'cot':
            prompts_list = create_several_input_prompts(args, cot_flag=True)
        elif args.method == 'standard':
            prompts_list = create_several_input_prompts(args, cot_flag=False)

        if args.model_id.startswith("gpt-35"):
            prompts_list = [create_prompt_template_gpt35(prompt, args) for prompt in prompts_list]
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
            y.append(line["final_answer"]) # use final_answer instead of pred_ans

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

def create_prompt_template_gpt35(prompt: str, args):
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

def initialize_opensource_llm():
    device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

    model = AutoModelForCausalLM.from_pretrained(
        'mosaicml/mpt-7b-instruct',
        trust_remote_code=True,
        torch_dtype=bfloat16,
        max_seq_len=1048
    )
    model.eval()
    model.to(device)

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    stop_token_ids = tokenizer.convert_tokens_to_ids(["<|endoftext|>"])

    class StopOnTokens(StoppingCriteria):
        def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
            for stop_id in stop_token_ids:
                if input_ids[0][-1] == stop_id:
                    return True
            return False

    stopping_criteria = StoppingCriteriaList([StopOnTokens()])

    pipeline_text_generation = pipeline(
                model=model, tokenizer=tokenizer,
                return_full_text=True,  # langchain expects the full text
                task='text-generation',
                device=device,
                stopping_criteria=stopping_criteria, 
                temperature=0.0,
                max_new_tokens=1024,  
            )

    llm = HuggingFacePipeline(pipeline=pipeline_text_generation)
    return llm


def initialize_llm(args, is_azureopenai=True):
    if is_azureopenai:

        headers = create_header_llm()

        if args.model_id.startswith("gpt-35"):
            llm = AzureChatOpenAI(
            deployment_name=args.model_id,
            model_name=args.model_id,
            temperature=args.temperature,
            headers=headers,
            max_tokens=1024,
            openai_api_base=OPENAI_API_BASE,
            openai_api_type=OPENAI_API_TYPE,
            openai_api_version=OPENAI_API_VERSION,
            openai_api_key=AZURE_OPENAI_API_KEY
            )
        else:
            if args.model_id.startswith('text-davinci'):
                llm = AzureOpenAI(
                deployment_name=args.model_id,
                model_name=args.model_id,
                temperature=args.temperature,
                headers=headers,
                max_tokens=1024,
                )
            else:
                pass
                # if args.model_id == 'mosaicml/mpt-7b-instruct':
                #     device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

                #     model = AutoModelForCausalLM.from_pretrained(
                #         args.model_id,
                #         trust_remote_code=True,
                #         torch_dtype=bfloat16,
                #         max_seq_len=2048
                #     )
                #     model.eval()
                #     model.to(device)

                #     tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
                #     stop_token_ids = tokenizer.convert_tokens_to_ids(["<|endoftext|>"])

                #     class StopOnTokens(StoppingCriteria):
                #         def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
                #             for stop_id in stop_token_ids:
                #                 if input_ids[0][-1] == stop_id:
                #                     return True
                #             return False

                #     stopping_criteria = StoppingCriteriaList([StopOnTokens()])

                #     pipeline_text_generation = pipeline(
                #                 model=model, tokenizer=tokenizer,
                #                 return_full_text=True,  # langchain expects the full text
                #                 task='text-generation',
                #                 device=device,
                #                 stopping_criteria=stopping_criteria, 
                #                 temperature=args.temperature,
                #                 max_new_tokens=1024,  
                #             )

                # elif args.model_id == 'tiiuae/falcon-40b-instruct':
                #     tokenizer = AutoTokenizer.from_pretrained(args.model_id)
                #     model = AutoModelForCausalLM.from_pretrained(
                #         args.model_id,
                #         torch_dtype=torch.bfloat16,
                #         trust_remote_code=True,
                #         load_in_8bit=True, # load in 8bit mode to save memory
                #         device_map="auto"
                #     )

                #     pipeline_text_generation = pipeline(
                #         "text-generation",
                #         model=model,
                #         tokenizer=tokenizer,
                #     )
                # else:
                #     raise NotImplementedError(f"Model {args.model_id} not implemented")
                
                # llm = HuggingFacePipeline(pipeline=pipeline_text_generation)

                        # llm = HuggingFacePipeline.from_model_id(
                        # model_id=args.model_id,
                        # model_kwargs={"temperature": args.temperature,
                        #               "trust_remote_code": True,
                        #               "max_seq_len": 4096}
                        # )
    else:
        llm = ChatOpenAI(
                model='gpt-3.5-turbo-0613',
                temperature=args.temperature,
                max_tokens=1024,
                openai_api_key=OPENAI_API_KEY
                )
    return llm

# def initialize_llmchain(args, prompt_template, llm_init=False):
#     if not llm_init:
#         args.llm = initialize_llm(args)

#     args.llm_chain = LLMChain(prompt=prompt_template, llm=args.llm, verbose=False)

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

