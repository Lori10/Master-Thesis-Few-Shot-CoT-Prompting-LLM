import torch 
from transformers import StoppingCriteria, StoppingCriteriaList, AutoTokenizer, AutoModelForCausalLM, pipeline
from torch import cuda, bfloat16
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate

model_id = "mosaicml/mpt-7b-instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    torch_dtype=bfloat16,
    device_map="auto"
)


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
             return_full_text=True,
             task='text-generation',
             device_map="auto",
             stopping_criteria=stopping_criteria,
             do_sample=False,
             max_new_tokens=1000,
	     use_cache=True
         )

llm = HuggingFacePipeline(pipeline=pipeline_text_generation, model_id=model_id,
                          pipeline_kwargs={"return_full_text": True, "max_new_tokens": 1000})

print(llm)
