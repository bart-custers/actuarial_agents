from dotenv import load_dotenv
load_dotenv()   # this loads .env variables into environment
import os
from google.colab import drive
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain.llms import HuggingFacePipeline

drive.mount("/content/drive", force_remount=False)
model_cache_dir = "/content/drive/MyDrive/Thesis/model_cache"
os.makedirs(model_cache_dir, exist_ok=True)

from dotenv import load_dotenv
load_dotenv("/content/drive/MyDrive/Thesis/.env")
hf_token = os.getenv("HF_TOKEN")

class LLMWrapper:
    def __init__(
        self,
        backend="mock",
        llm=None,
        openai_model="gpt-4o-mini",
        openai_api_key=None,
        hf_token=None,
    ):
        """
        backend: one of ["openai", "phi3mini", "llama7b", "mock"]
        """
        self.backend = backend
        self.hf_token = hf_token or os.getenv("HF_TOKEN")

        if backend == "openai":
            self.llm = ChatOpenAI(
                model=openai_model,
                temperature=0.1,
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
        """Ensure all models return a string."""
        if self.backend == "openai":
            msg = HumanMessage(content=prompt)
            response = self.llm([msg])
            return response.content
        elif self.backend in ["phi3mini", "llama7b"]:
            return self.llm(prompt)
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

        print(f"Loading {model_name} ... (using cache at {model_path})")

        tokenizer = AutoTokenizer.from_pretrained(
            model_path, use_auth_token=self.hf_token, cache_dir=model_path
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype="auto",
            load_in_4bit=True,
            offload_folder="offload",
            use_auth_token=self.hf_token,
            cache_dir=model_path,
        )

        text_gen = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,
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

        print(f"Loading {model_name} ... (using cache at {model_path})")

        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            use_auth_token=self.hf_token,
            cache_dir=model_path,
            use_fast=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype="auto",
            load_in_4bit=True,
            use_auth_token=self.hf_token,
            cache_dir=model_path,
        )

        text_gen = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,
            do_sample=False,
        )

        hf_llm = HuggingFacePipeline(pipeline=text_gen)
        return lambda prompt: hf_llm.invoke(prompt)