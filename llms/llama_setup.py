from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain.llms import HuggingFacePipeline

HF_TOKEN = "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"

def get_llama7b_llm():
    model_name = "meta-llama/Llama-2-7b-chat-hf"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=HF_TOKEN)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        load_in_4bit=True,
        torch_dtype="auto",
        use_auth_token=HF_TOKEN
    )
    text_gen = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        do_sample=False  # deterministic
    )
    llm = HuggingFacePipeline(pipeline=text_gen)
    return llm

