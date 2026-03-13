import os
import torch
import numpy as np
from math import exp
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_community.llms import HuggingFacePipeline
from google.colab import drive
from dotenv import load_dotenv
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)

# ------------------------------------------------------------
# Environment & cache
# ------------------------------------------------------------
drive.mount("/content/drive", force_remount=False)

model_cache_dir = "/content/drive/MyDrive/Thesis/model_cache"
os.makedirs(model_cache_dir, exist_ok=True)

os.environ["HF_HOME"] = model_cache_dir
os.environ["HF_DATASETS_CACHE"] = model_cache_dir
os.environ["TRANSFORMERS_CACHE"] = model_cache_dir

load_dotenv("/content/drive/MyDrive/Thesis/.env")
hf_token = os.getenv("HF_TOKEN")

# ------------------------------------------------------------
# LLM Wrapper
# ------------------------------------------------------------
class LLMWrapper:
    def __init__(
        self,
        backend="llama7b",
        system_prompt=None,
        hf_token=None,
    ):
        """
        backend: one of ["llama7b", "llama31_8b", "qwen25_7b", "mock"]
        """

        self.backend = backend
        self.hf_token = hf_token or os.getenv("HF_TOKEN")
        self.system_prompt = (
            system_prompt
            or "You are a helpful assistant that writes concise, professional explanations."
        )

        if backend == "llama7b":
            self._init_llama7b()

        elif backend == "llama31_8b":
            self._init_llama31_8b()

        elif backend == "qwen25_7b":
            self._init_qwen25_7b()

        elif backend == "mock":
            self.llm = lambda prompt: '{"mock_output": "simulated response"}'

        else:
            raise ValueError(f"Unknown backend '{backend}'")

    # ------------------------------------------------------------
    # Unified callable interface
    # ------------------------------------------------------------
    def __call__(self, prompt, return_uncertainty=False):
        if self.backend == "mock":
            text = self.llm(prompt)
            if return_uncertainty:
                return text, {"method": "mock"}
            return text

        if return_uncertainty:
            out = self.generate_with_logprobs(prompt)
            uncertainty = exp(out["mean_logprob"])
            return out["text"], uncertainty

        formatted = self._format_prompt(prompt)
        raw = self.llm(formatted)

        if isinstance(raw, str):
            return raw
        elif isinstance(raw, list) and "generated_text" in raw[0]:
            return raw[0]["generated_text"]
        else:
            return str(raw)

    # ------------------------------------------------------------
    # Prompt formatting (model-specific)
    # ------------------------------------------------------------
    def _format_prompt(self, prompt):
        if self.backend == "llama7b":
            return (
                f"<s>[INST] <<SYS>>\n{self.system_prompt}\n<</SYS>>\n"
                f"{prompt}\n[/INST]"
            )

        if self.backend in ["llama31_8b", "qwen25_7b"]:
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ]
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        return prompt

    # ------------------------------------------------------------
    # Logprob-enabled generation (ALL HF models)
    # ------------------------------------------------------------
    def generate_with_logprobs(
        self,
        prompt,
        max_new_tokens=1024,
        temperature=0.7,
        do_sample=True,
    ):
        device = next(self.model.parameters()).device
        formatted = self._format_prompt(prompt)

        inputs = self.tokenizer(
            formatted, return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                return_dict_in_generate=True,
                output_scores=True,
            )

        gen_ids = out.sequences[0][inputs["input_ids"].shape[1]:]
        scores = out.scores

        tokens = []
        joint_logprob = 0.0

        for tid, score in zip(gen_ids, scores):
            log_probs = torch.log_softmax(score[0], dim=-1)
            lp = log_probs[tid].item()

            tokens.append({
                "token": self.tokenizer.decode(tid),
                "logprob": lp,
                "prob": exp(lp),
            })

            joint_logprob += lp

        return {
            "text": self.tokenizer.decode(gen_ids, skip_special_tokens=True),
            "tokens": tokens,
            "joint_logprob": joint_logprob,
            "mean_logprob": joint_logprob / max(len(tokens), 1),
        }

    # ------------------------------------------------------------
    # Model initializers
    # ------------------------------------------------------------
    def _init_llama7b(self):
        model_name = "meta-llama/Llama-2-7b-chat-hf"
        self._init_hf_model(model_name)

    def _init_llama31_8b(self):
        model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
        self._init_hf_model(model_name)

    def _init_qwen25_7b(self):
        model_name = "Qwen/Qwen2.5-7B-Instruct"
        self._init_hf_model(model_name)

    def _init_hf_model(self, model_name):
        model_path = os.path.join(model_cache_dir, model_name.replace("/", "_"))
        os.makedirs(model_path, exist_ok=True)

        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            token=self.hf_token,
            cache_dir=model_path,
            trust_remote_code=True        # Needed for custom model code
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # --------------------------
        # Model
        # --------------------------
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            #load_in_4bit=True,
            torch_dtype="auto",
            token=self.hf_token,
            cache_dir=model_path,
            trust_remote_code=True
        )

        model.eval()

        text_gen = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=1024,
            do_sample=True,
            temperature=0.7,
            truncation=False,
            trust_remote_code=True
        )

        self.model = model
        self.tokenizer = tokenizer
        self.llm = HuggingFacePipeline(pipeline=text_gen).invoke

    # ------------------------------------------------------------
    # Hidden states (TCAV-compatible)
    # ------------------------------------------------------------
    def get_hidden_states_for_texts(self, texts, layer=-1, pooling="mean"):
        if not hasattr(self, "model") or not hasattr(self, "tokenizer"):
            raise RuntimeError("Model/tokenizer not loaded")

        device = next(self.model.parameters()).device
        vectors = []

        with torch.no_grad():
            for text in texts:
                toks = self.tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                ).to(device)

                out = self.model(
                    **toks,
                    output_hidden_states=True,
                )

                hs = out.hidden_states[layer]  # (1, seq, dim)

                if pooling == "mean":
                    vec = hs.mean(dim=1).squeeze(0)
                elif pooling == "max":
                    vec = hs.max(dim=1).values.squeeze(0)
                elif pooling == "last":
                    vec = hs[:, -1, :].squeeze(0)
                else:
                    raise ValueError("pooling must be one of ['mean', 'max', 'last']")

                #vectors.append(vec.cpu().numpy())
                vectors.append(vec.detach().cpu().float().numpy())

        return np.array(vectors)