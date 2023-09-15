from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import transformers
import torch
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

model_id = "vilsonrodrigues/falcon-7b-instruct-sharded"

model_4bit = AutoModelForCausalLM.from_pretrained(
        model_id, 
        device_map="auto",
        quantization_config=quantization_config,
        )

tokenizer = AutoTokenizer.from_pretrained(model_id)

pipeline = pipeline(
        "text-generation",
        model=model_4bit,
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

print(pipeline("Who is Lionel Messi?"))
