from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import transformers
import torch
from transformers import BitsAndBytesConfig
from langchain.llms import HuggingFacePipeline

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

model_id = "vilsonrodrigues/falcon-7b-instruct-sharded"

model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        quantization_config=quantization_config,
        )

tokenizer = AutoTokenizer.from_pretrained(model_id)
pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        use_cache=True,
        device_map="auto",
        max_length=296,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
)

llm = HuggingFacePipeline(pipeline=pipe, model_id=model_id, model_kwargs={"quantization_config": quantization_config}, pipeline_kwargs={ "return_full_text":True})

print(llm("Who is Lionel Messi?"))
