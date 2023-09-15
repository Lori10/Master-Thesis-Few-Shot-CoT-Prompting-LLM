from transformers import AutoTokenizer
import transformers
import torch
from torch import cuda, bfloat16
from langchain.llms import HuggingFacePipeline

model_id = "tiiuae/falcon-7b-instruct"
device = f"cuda:{cuda.current_device()}" if cuda.is_available() else "cpu"

llm = HuggingFacePipeline.from_model_id(
	model_id=model_id,
	device=cuda.current_device(),
	task="text-generation",
	model_kwargs={"trust_remote_code": True},
        pipeline_kwargs={"return_full_text": True, "temperature": 0.0}
)

print(llm("Who is Lionel Messi?"))
