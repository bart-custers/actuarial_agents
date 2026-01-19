import os
import torch
import numpy as np
from math import exp
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_community.llms import HuggingFacePipeline
from google.colab import drive
from dotenv import load_dotenv

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
        backend: one of ["llama7b", "llama31_8b", "mistral3_14b", "mock"]
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

        elif backend == "mistral3_14b":
            self._init_mistral3_14b()

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

        if self.backend == "llama31_8b":
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ]
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        if self.backend == "mistral3_14b":
            return f"<s>[INST] {self.system_prompt}\n\n{prompt} [/INST]"

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

    def _init_mistral3_14b(self):
        model_name = "mistralai/Ministral-3-14B-Reasoning-2512"
        self._init_hf_model(model_name)

    def _init_hf_model(self, model_name):
        model_path = os.path.join(model_cache_dir, model_name.replace("/", "_"))
        os.makedirs(model_path, exist_ok=True)

        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            token=self.hf_token,
            cache_dir=model_path,
            use_fast=True,
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            load_in_4bit=True,
            torch_dtype="auto",
            token=self.hf_token,
            cache_dir=model_path,
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

                vectors.append(vec.cpu().numpy())

        return np.array(vectors)

# ------------------------------------------------------------



# import os
# from google.colab import drive
# from langchain_openai import ChatOpenAI
# from langchain_core.messages import HumanMessage
# from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
# from langchain_community.llms import HuggingFacePipeline
# import torch
# import numpy as np
# from math import exp

# drive.mount("/content/drive", force_remount=False)
# model_cache_dir = "/content/drive/MyDrive/Thesis/model_cache"
# os.makedirs(model_cache_dir, exist_ok=True)

# from dotenv import load_dotenv
# load_dotenv("/content/drive/MyDrive/Thesis/.env")
# hf_token = os.getenv("HF_TOKEN")

# os.environ["HF_HOME"] = "/content/drive/MyDrive/Thesis/model_cache"
# os.environ["HF_DATASETS_CACHE"] = "/content/drive/MyDrive/Thesis/model_cache"
# os.environ["TRANSFORMERS_CACHE"] = "/content/drive/MyDrive/Thesis/model_cache"

# class LLMWrapper:
#     def __init__(
#         self,
#         backend="mock",
#         llm=None,
#         openai_model="gpt-4o-mini",
#         openai_api_key=None,
#         hf_token=None,
#         system_prompt=None,
#     ):
#         """
#         backend: one of ["openai", "phi3mini", "llama7b", "mock"]
#         """
#         self.backend = backend
#         self.system_prompt = (
#             system_prompt
#             or "You are a helpful assistant that writes concise, professional explanations."
#         )
#         self.hf_token = hf_token or os.getenv("HF_TOKEN")

#         if backend == "openai":
#             self.llm = ChatOpenAI(
#                 model=openai_model,
#                 temperature=0.7,
#                 max_tokens=512,
#                 openai_api_key=openai_api_key or os.getenv("OPENAI_API_KEY"),
#             )

#         elif backend == "phi3mini":
#             self.llm = self._init_phi3_mini()

#         elif backend == "llama7b":
#             self.llm = self._init_llama7b()

#         elif backend == "mock":
#             self.llm = llm or (lambda prompt: '{"mock_output": "simulated response"}')

#         else:
#             raise ValueError(f"Unknown backend '{backend}'")

#     # ------------------------------------------------------------
#     # Unified callable interface
#     # ------------------------------------------------------------
#     def __call__(self, prompt, return_uncertainty=False):
#         """
#         Default: returns text only
#         Optional: returns (text, uncertainty_dict)
#         """

#         # -------------------------
#         # OPENAI (no logprobs here)
#         # -------------------------
#         if self.backend == "openai":
#             msg = HumanMessage(content=prompt)
#             response = self.llm([msg])
#             text = response.content

#             if return_uncertainty:
#                 return text, {
#                     "method": "none",
#                     "note": "logprobs not enabled for OpenAI backend"
#                 }

#             return text

#         # -------------------------
#         # LLAMA / PHI (HF backends)
#         # -------------------------
#         elif self.backend in ["llama7b", "phi3mini"]:

#             if return_uncertainty and self.backend == "llama7b":
#                 out = self.generate_with_logprobs(prompt)

#                 uncertainty = exp(out["mean_logprob"])

#                 return out["text"], uncertainty

#             # fallback: text only
#             raw = self.llm(prompt)

#             if isinstance(raw, dict) and "generated_text" in raw:
#                 return raw["generated_text"]
#             elif isinstance(raw, list) and "generated_text" in raw[0]:
#                 return raw[0]["generated_text"]
#             elif hasattr(raw, "generations"):
#                 return raw.generations[0][0].text
#             else:
#                 return str(raw)

#         # -------------------------
#         # MOCK
#         # -------------------------
#         elif self.backend == "mock":
#             text = self.llm(prompt)
#             if return_uncertainty:
#                 return text, {"method": "mock"}
#             return text

#         else:
#             raise ValueError(f"Unsupported backend: {self.backend}")


#     # ------------------------------------------------------------
#     # Hugging Face: Phi-3 Mini (3.8B)
#     # ------------------------------------------------------------
#     def _init_phi3_mini(self):
#         model_name = "microsoft/phi-3-mini-128k-instruct"
#         model_path = os.path.join(model_cache_dir, model_name.replace("/", "_"))
#         os.makedirs(model_path, exist_ok=True)

#         tokenizer = AutoTokenizer.from_pretrained(
#             model_name, token=self.hf_token
#         )
#         model = AutoModelForCausalLM.from_pretrained(
#             model_name,
#             device_map="auto",
#             dtype="auto",
#             load_in_4bit=True,
#             offload_folder="offload",
#             token=self.hf_token,
#             #cache_dir=model_path,
#         )

#         text_gen = pipeline(
#             "text-generation",
#             model=model,
#             tokenizer=tokenizer,
#             max_new_tokens=1600,
#             do_sample=False,
#         )

#         hf_llm = HuggingFacePipeline(pipeline=text_gen)
#         return lambda prompt: hf_llm.invoke(prompt)

#     # ------------------------------------------------------------
#     # Hugging Face: LLaMA 7B
#     # ------------------------------------------------------------
#     def generate_with_logprobs(
#             self,
#             prompt,
#             max_new_tokens=1600,
#             temperature=0.7,
#             do_sample=True
#         ):
#             """
#             Returns:
#             {
#                 "text": str,
#                 "tokens": [
#                     {"token": str, "logprob": float, "prob": float}
#                 ],
#                 "joint_logprob": float,
#                 "mean_logprob": float
#             }
#             """

#             if self.backend != "llama7b":
#                 raise RuntimeError("generate_with_logprobs is only supported for backend='llama7b'")

#             device = next(self.model.parameters()).device

#             formatted_prompt = (
#                 f"<s>[INST] <<SYS>>\n{self.system_prompt}\n<</SYS>>\n"
#                 f"{prompt}\n[/INST]"
#             )

#             inputs = self.tokenizer(
#                 formatted_prompt,
#                 return_tensors="pt"
#             ).to(device)

#             with torch.no_grad():
#                 gen_out = self.model.generate(
#                     **inputs,
#                     max_new_tokens=max_new_tokens,
#                     do_sample=do_sample,
#                     temperature=temperature if do_sample else None,
#                     return_dict_in_generate=True,
#                     output_scores=True
#                 )

#             # Only newly generated tokens (exclude prompt)
#             gen_token_ids = gen_out.sequences[0][inputs["input_ids"].shape[1]:]

#             scores = gen_out.scores  # list: one [1, vocab] tensor per token

#             tokens = []
#             joint_logprob = 0.0

#             for token_id, score in zip(gen_token_ids, scores):
#                 log_probs = torch.log_softmax(score[0], dim=-1)
#                 token_logprob = log_probs[token_id].item()

#                 token_str = self.tokenizer.decode(token_id)

#                 tokens.append({
#                     "token": token_str,
#                     "logprob": token_logprob,
#                     "prob": exp(token_logprob)
#                 })

#                 joint_logprob += token_logprob

#             text = self.tokenizer.decode(
#                 gen_token_ids,
#                 skip_special_tokens=True
#             )

#             mean_logprob = joint_logprob / max(len(tokens), 1)

#             return {
#                 "text": text,
#                 "tokens": tokens,
#                 "joint_logprob": joint_logprob,
#                 "mean_logprob": mean_logprob
#             }
    
#     def _init_llama7b(self):
#         model_name = "meta-llama/Llama-2-7b-chat-hf"
#         model_path = os.path.join(model_cache_dir, model_name.replace("/", "_"))
#         os.makedirs(model_path, exist_ok=True)

#         tokenizer = AutoTokenizer.from_pretrained(
#             model_name,
#             token=self.hf_token,
#             cache_dir=model_path,
#             use_fast=True,
#         )

#         # NEW
#         if tokenizer.pad_token is None:
#             tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})

#         model = AutoModelForCausalLM.from_pretrained(
#             model_name,
#             device_map="auto",
#             dtype="auto",
#             load_in_4bit=True,
#             token=self.hf_token,
#             cache_dir=model_path,
#         )

#         # NEW
#         model.resize_token_embeddings(len(tokenizer))

#         self.tokenizer = tokenizer
#         self.model = model
#         self.model.eval()

#         text_gen = pipeline(
#             "text-generation",
#             model=model,
#             tokenizer=tokenizer,
#             max_new_tokens=1024,
#             truncation=False,
#             do_sample=True,          
#             temperature=0.7,         
#         )

#         hf_llm = HuggingFacePipeline(pipeline=text_gen)

#     # Add LLaMA chat-format wrapper
#         def llama_chat(prompt):
#             formatted_prompt = f"<s>[INST] <<SYS>>\n{self.system_prompt}\n<</SYS>>\n{prompt}\n[/INST]"
#             out = hf_llm.invoke(formatted_prompt)
#             # Extract text safely
#             if isinstance(out, str):
#                 return out
#             elif isinstance(out, list) and "generated_text" in out[0]:
#                 return out[0]["generated_text"]
#             else:
#                 return str(out)

#         return llama_chat
    
#     # ------------------------------------------------------------
#     # Hidden state extraction for TCAV
#     # ------------------------------------------------------------
#     def get_hidden_states_for_texts(self, texts, layer=20):
#         """
#         Returns: numpy array (N, hidden_dim)
#         Uses model output_hidden_states=True to retrieve internal activations.
#         """
#         if not hasattr(self, "model") or not hasattr(self, "tokenizer"):
#             raise RuntimeError("LLMWrapper has no model/tokenizer loaded (did you choose backend=llama7b?)")

#         hidden_vectors = []
#         device = next(self.model.parameters()).device

#         with torch.no_grad():
#             for text in texts:
#                 toks = self.tokenizer(
#                     text,
#                     return_tensors="pt",
#                     truncation=True,
#                     padding=True
#                 ).to(device)

#                 out = self.model(
#                     **toks,
#                     output_hidden_states=True
#                 )
#                 # hidden_states: tuple of length num_layers+1
#                 # each element: (1, seq_len, hidden_dim)
#                 hs = out.hidden_states[layer]      # layer
#                 sent_emb = hs.mean(dim=1).squeeze(0)   # mean-pool tokens
#                 hidden_vectors.append(sent_emb.cpu().numpy())

#         return np.array(hidden_vectors)