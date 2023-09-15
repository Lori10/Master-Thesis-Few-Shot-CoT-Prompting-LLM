
import torch
from transformers import StoppingCriteria, StoppingCriteriaList, AutoTokenizer, AutoModelForCausalLM, pipeline
from torch import cuda, bfloat16
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline, TextStreamer


device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'
model_id = 'Trelis/mpt-7b-instruct-hosted-inference-8bit'
print(device)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    load_in_8bit=True
)
model.eval()

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
stop_token_ids = tokenizer.convert_tokens_to_ids(["<|endoftext|>"])

class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for stop_id in stop_token_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False

stopping_criteria = StoppingCriteriaList([StopOnTokens()])


llm = HuggingFacePipeline.from_model_id(
                model_id=model_id,
                task="text-generation",
                model_kwargs={"trust_remote_code": True,
                             "max_seq_len": 1048,
                             "load_in_8bit": True,
                             "device_map": "auto"
                           },
                pipeline_kwargs={
                      "return_full_text":True,
                      "stopping_criteria": stopping_criteria,
                      "temperature": 0.0,
                     "max_new_tokens": 1024
    }
  )
