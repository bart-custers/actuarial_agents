import os
from google.colab import drive
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_community.llms import HuggingFacePipeline

drive.mount("/content/drive", force_remount=False)
model_cache_dir = "/content/drive/MyDrive/Thesis/model_cache"
os.makedirs(model_cache_dir, exist_ok=True)

from dotenv import load_dotenv
load_dotenv("/content/drive/MyDrive/Thesis/.env")
hf_token = os.getenv("HF_TOKEN")

os.environ["HF_HOME"] = "/content/drive/MyDrive/Thesis/model_cache"
os.environ["HF_DATASETS_CACHE"] = "/content/drive/MyDrive/Thesis/model_cache"
os.environ["TRANSFORMERS_CACHE"] = "/content/drive/MyDrive/Thesis/model_cache"

class LLMWrapper:
    def __init__(
        self,
        backend="mock",
        llm=None,
        openai_model="gpt-4o-mini",
        openai_api_key=None,
        hf_token=None,
        system_prompt=None,
    ):
        """
        backend: one of ["openai", "phi3mini", "llama7b", "mock"]
        """
        self.backend = backend
        self.system_prompt = (
            system_prompt
            or "You are a helpful assistant that writes concise, professional explanations."
        )
        self.hf_token = hf_token or os.getenv("HF_TOKEN")

        if backend == "openai":
            self.llm = ChatOpenAI(
                model=openai_model,
                temperature=0.7,
                max_tokens=512,
                openai_api_key=openai_api_key or os.getenv("OPENAI_API_KEY"),
            )

        elif backend == "phi3mini":
            self.llm = self._init_phi3_mini()

        elif backend == "llama7b":
            self.llm = self._init_llama7b()

        elif backend == "mock":
            self.llm = llm or (lambda prompt: '{"mock_output": "simulated response"}')

        else:
            raise ValueError(f"Unknown backend '{backend}'")

    # ------------------------------------------------------------
    # Unified callable interface
    # ------------------------------------------------------------
    def __call__(self, prompt):
        """Ensure all models return a plain string output."""
        if self.backend == "openai":
            msg = HumanMessage(content=prompt)
            response = self.llm([msg])
            return response.content

        elif self.backend in ["phi3mini", "llama7b"]:
            raw = self.llm(prompt)
            # ✅ Normalize Hugging Face / LangChain output formats
            if isinstance(raw, dict) and "generated_text" in raw:
                return raw["generated_text"]
            elif isinstance(raw, list) and isinstance(raw[0], dict) and "generated_text" in raw[0]:
                return raw[0]["generated_text"]
            elif hasattr(raw, "generations"):  # LangChain LLMResult
                return raw.generations[0][0].text
            elif hasattr(raw, "content"):
                return raw.content
            else:
                return str(raw)

        elif self.backend == "mock":
            return self.llm(prompt)

        else:
            raise ValueError(f"Unsupported backend: {self.backend}")

    # ------------------------------------------------------------
    # Hugging Face: Phi-3 Mini (3.8B)
    # ------------------------------------------------------------
    def _init_phi3_mini(self):
        model_name = "microsoft/phi-3-mini-128k-instruct"
        model_path = os.path.join(model_cache_dir, model_name.replace("/", "_"))
        os.makedirs(model_path, exist_ok=True)

        #print(f"Loading {model_name} ... (using cache at {model_path})")

        tokenizer = AutoTokenizer.from_pretrained(
            model_name, token=self.hf_token
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            dtype="auto",
            load_in_4bit=True,
            offload_folder="offload",
            token=self.hf_token,
            #cache_dir=model_path,
        )

        text_gen = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=1024,
            do_sample=False,
        )

        hf_llm = HuggingFacePipeline(pipeline=text_gen)
        return lambda prompt: hf_llm.invoke(prompt)

    # ------------------------------------------------------------
    # Hugging Face: LLaMA 7B
    # ------------------------------------------------------------
    def _init_llama7b(self):
        model_name = "meta-llama/Llama-2-7b-chat-hf"
        model_path = os.path.join(model_cache_dir, model_name.replace("/", "_"))
        os.makedirs(model_path, exist_ok=True)

        #print(f"Loading {model_name} ... (using cache at {model_path})")

        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            token=self.hf_token,
            cache_dir=model_path,
            use_fast=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            dtype="auto",
            load_in_4bit=True,
            token=self.hf_token,
            cache_dir=model_path,
        )

        text_gen = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=1024,
            truncation=False,
            do_sample=True,          
            temperature=0.7,         
        )

        hf_llm = HuggingFacePipeline(pipeline=text_gen)

    # Add LLaMA chat-format wrapper
        def llama_chat(prompt):
            formatted_prompt = f"<s>[INST] <<SYS>>\n{self.system_prompt}\n<</SYS>>\n{prompt}\n[/INST]"
            out = hf_llm.invoke(formatted_prompt)
            # Extract text safely
            if isinstance(out, str):
                return out
            elif isinstance(out, list) and "generated_text" in out[0]:
                return out[0]["generated_text"]
            else:
                return str(out)

        return llama_chat