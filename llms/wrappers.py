import os
from google.colab import drive
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_community.llms import HuggingFacePipeline
import torch
import numpy as np

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
    # def __call__(self, prompt):
    #     """Ensure all models return a plain string output."""
    #     if self.backend == "openai":
    #         msg = HumanMessage(content=prompt)
    #         response = self.llm([msg])
    #         return response.content

    #     elif self.backend in ["phi3mini", "llama7b"]:
    #         raw = self.llm(prompt)
    #         # ✅ Normalize Hugging Face / LangChain output formats
    #         if isinstance(raw, dict) and "generated_text" in raw:
    #             return raw["generated_text"]
    #         elif isinstance(raw, list) and isinstance(raw[0], dict) and "generated_text" in raw[0]:
    #             return raw[0]["generated_text"]
    #         elif hasattr(raw, "generations"):  # LangChain LLMResult
    #             return raw.generations[0][0].text
    #         elif hasattr(raw, "content"):
    #             return raw.content
    #         else:
    #             return str(raw)

    #     elif self.backend == "mock":
    #         return self.llm(prompt)

    #     else:
    #         raise ValueError(f"Unsupported backend: {self.backend}")

    def __call__(self, prompt, return_uncertainty=False):
        """
        Default: returns text only
        Optional: returns (text, uncertainty_dict)
        """

        # -------------------------
        # OPENAI (no logprobs here)
        # -------------------------
        if self.backend == "openai":
            msg = HumanMessage(content=prompt)
            response = self.llm([msg])
            text = response.content

            if return_uncertainty:
                return text, {
                    "method": "none",
                    "note": "logprobs not enabled for OpenAI backend"
                }

            return text

        # -------------------------
        # LLAMA / PHI (HF backends)
        # -------------------------
        elif self.backend in ["llama7b", "phi3mini"]:

            if return_uncertainty and self.backend == "llama7b":
                out = self.generate_with_logprobs(prompt)

                uncertainty = {
                    "method": "token_logprob",
                    "joint_logprob": out["joint_logprob"],
                    "mean_logprob": out["mean_logprob"],
                    "n_tokens": len(out["tokens"])
                }

                return out["text"], uncertainty

            # fallback: text only
            raw = self.llm(prompt)

            if isinstance(raw, dict) and "generated_text" in raw:
                return raw["generated_text"]
            elif isinstance(raw, list) and "generated_text" in raw[0]:
                return raw[0]["generated_text"]
            elif hasattr(raw, "generations"):
                return raw.generations[0][0].text
            else:
                return str(raw)

        # -------------------------
        # MOCK
        # -------------------------
        elif self.backend == "mock":
            text = self.llm(prompt)
            if return_uncertainty:
                return text, {"method": "mock"}
            return text

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
    from math import exp

    def generate_with_logprobs(
            self,
            prompt,
            max_new_tokens=128,
            temperature=0.7,
            do_sample=True
        ):
            """
            Returns:
            {
                "text": str,
                "tokens": [
                    {"token": str, "logprob": float, "prob": float}
                ],
                "joint_logprob": float,
                "mean_logprob": float
            }
            """

            if self.backend != "llama7b":
                raise RuntimeError("generate_with_logprobs is only supported for backend='llama7b'")

            device = next(self.model.parameters()).device

            formatted_prompt = (
                f"<s>[INST] <<SYS>>\n{self.system_prompt}\n<</SYS>>\n"
                f"{prompt}\n[/INST]"
            )

            inputs = self.tokenizer(
                formatted_prompt,
                return_tensors="pt"
            ).to(device)

            with torch.no_grad():
                gen_out = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    temperature=temperature if do_sample else None,
                    return_dict_in_generate=True,
                    output_scores=True
                )

            # Only newly generated tokens (exclude prompt)
            gen_token_ids = gen_out.sequences[0][inputs["input_ids"].shape[1]:]

            scores = gen_out.scores  # list: one [1, vocab] tensor per token

            tokens = []
            joint_logprob = 0.0

            for token_id, score in zip(gen_token_ids, scores):
                log_probs = torch.log_softmax(score[0], dim=-1)
                token_logprob = log_probs[token_id].item()

                token_str = self.tokenizer.decode(token_id)

                tokens.append({
                    "token": token_str,
                    "logprob": token_logprob,
                    "prob": exp(token_logprob)
                })

                joint_logprob += token_logprob

            text = self.tokenizer.decode(
                gen_token_ids,
                skip_special_tokens=True
            )

            mean_logprob = joint_logprob / max(len(tokens), 1)

            return {
                "text": text,
                "tokens": tokens,
                "joint_logprob": joint_logprob,
                "mean_logprob": mean_logprob
            }
    
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

        # NEW
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            dtype="auto",
            load_in_4bit=True,
            token=self.hf_token,
            cache_dir=model_path,
        )

        # NEW
        model.resize_token_embeddings(len(tokenizer))

        self.tokenizer = tokenizer
        self.model = model
        self.model.eval()

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
    
    # ------------------------------------------------------------
    # Hidden state extraction for TCAV
    # ------------------------------------------------------------
    def get_hidden_states_for_texts(self, texts, layer=20):
        """
        Returns: numpy array (N, hidden_dim)
        Uses model output_hidden_states=True to retrieve internal activations.
        """
        if not hasattr(self, "model") or not hasattr(self, "tokenizer"):
            raise RuntimeError("LLMWrapper has no model/tokenizer loaded (did you choose backend=llama7b?)")

        hidden_vectors = []
        device = next(self.model.parameters()).device

        with torch.no_grad():
            for text in texts:
                toks = self.tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    padding=True
                ).to(device)

                out = self.model(
                    **toks,
                    output_hidden_states=True
                )
                # hidden_states: tuple of length num_layers+1
                # each element: (1, seq_len, hidden_dim)
                hs = out.hidden_states[layer]      # layer
                sent_emb = hs.mean(dim=1).squeeze(0)   # mean-pool tokens
                hidden_vectors.append(sent_emb.cpu().numpy())

        return np.array(hidden_vectors)