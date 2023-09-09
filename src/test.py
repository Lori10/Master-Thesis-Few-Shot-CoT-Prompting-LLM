import torch 
from transformers import StoppingCriteria, StoppingCriteriaList, AutoTokenizer, AutoModelForCausalLM, pipeline
from torch import cuda, bfloat16
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate

device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'
print(device)

model = AutoModelForCausalLM.from_pretrained(
    'mosaicml/mpt-7b-instruct',
    trust_remote_code=True,
    torch_dtype=bfloat16,
    max_seq_len=1048
)
model.eval()
model.to(device)

# tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
# stop_token_ids = tokenizer.convert_tokens_to_ids(["<|endoftext|>"])

# class StopOnTokens(StoppingCriteria):
#     def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
#         for stop_id in stop_token_ids:
#             if input_ids[0][-1] == stop_id:
#                 return True
#         return False

# stopping_criteria = StoppingCriteriaList([StopOnTokens()])

# pipeline_text_generation = pipeline(
#             model=model, tokenizer=tokenizer,
#             return_full_text=True,  # langchain expects the full text
#             task='text-generation',
#             device=device,
#             stopping_criteria=stopping_criteria, 
#             temperature=0.0,
#             max_new_tokens=1024,  
#         )

# llm = HuggingFacePipeline(pipeline=pipeline_text_generation)

