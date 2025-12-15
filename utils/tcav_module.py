import os
import json
from typing import List, Dict, Tuple
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd

try:
    from sklearn.svm import LinearSVC
except:
    raise ImportError("scikit-learn required for LinearSVC (pip install scikit-learn)")

# Layer extractor
class LLMLayerExtractor:
    """
    Adapter to extract hidden states from:
    - The user's LLMWrapper (preferred)
    - Or a raw HuggingFace transformers model (fallback)
    """

    def __init__(self, llm_wrapper=None, model_name=None, device=None):
        self.llm_wrapper = llm_wrapper
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def _batch(self, items: List[str], batch_size: int):
        for i in range(0, len(items), batch_size):
            yield items[i:i+batch_size]

    def get_hidden_embeddings(self, texts: List[str], layer: int, batch_size: int = 8) -> np.ndarray:
        """
        Returns (N, hidden_dim) array of embeddings, one per text.
        Works for wrappers that collapse batches into a single vector.
        """
        out = []

        if self.llm_wrapper is not None:

            if hasattr(self.llm_wrapper, "get_hidden_states_for_texts"):
                for batch in self._batch(texts, batch_size):
                    emb_batch = self.llm_wrapper.get_hidden_states_for_texts(batch, layer)
                    if emb_batch.shape[0] == 1 and len(batch) > 1:
                        emb_batch = np.repeat(emb_batch, len(batch), axis=0)
                    out.append(emb_batch)
                return np.vstack(out)

            if hasattr(self.llm_wrapper, "embed_texts_by_layer"):
                for batch in self._batch(texts, batch_size):
                    arr = np.asarray(self.llm_wrapper.embed_texts_by_layer(batch, layer))
                    if arr.shape[0] == 1 and len(batch) > 1:
                        arr = np.repeat(arr, len(batch), axis=0)
                    out.append(arr)
                return np.vstack(out)

        if hasattr(self, "use_transformers") and self.use_transformers:
            vecs = []
            with torch.no_grad():
                for batch in self._batch(texts, batch_size):
                    toks = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(self.device)
                    out_model = self.model(**toks)
                    hidden = out_model.hidden_states[layer]
                    sent = hidden.mean(dim=1).cpu().numpy()
                    vecs.append(sent)
            return np.vstack(vecs)

        raise RuntimeError("No hidden state extraction available.")

# Train CAV
def train_cav(concept_embs: np.ndarray, random_embs: np.ndarray,
              C=0.1, max_iter=50000) -> Tuple[np.ndarray, dict]:
    X = np.vstack([concept_embs, random_embs])
    y = np.hstack([np.ones(len(concept_embs)), np.zeros(len(random_embs))])
    clf = LinearSVC(C=C, max_iter=max_iter)
    clf.fit(X, y)
    cav = clf.coef_.reshape(-1)
    norm = np.linalg.norm(cav)
    if norm > 0:
        cav = cav / norm
    meta = {"C": C, "max_iter": max_iter, "n_samples": len(X)}
    return cav, {"clf": clf, "meta": meta}

# Save and load CAV
def save_cav(path: str, cav: np.ndarray, meta: dict, clf=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path + ".npy", cav)
    with open(path + ".meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    if clf is not None:
        import joblib
        joblib.dump(clf, path + ".joblib")

def load_cav(path: str) -> Tuple[np.ndarray, dict]:
    cav = np.load(path + ".npy")
    meta = {}
    meta_path = path + ".meta.json"
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta = json.load(f)
    clf = None
    clf_path = path + ".joblib"
    if os.path.exists(clf_path):
        import joblib
        clf = joblib.load(clf_path)
    return cav, {"meta": meta, "clf": clf}

# TCAV analysis
def directional_derivatives(embs: np.ndarray, cav: np.ndarray) -> np.ndarray:
    return embs.dot(cav)

def tcav_score(dots: np.ndarray) -> float:
    return float((dots > 0).mean())

def debug_tcav(dots, label):
    print(f"\n[TCAV DEBUG] {label}")
    print(f"  min={np.min(dots):.4f}")
    print(f"  mean={np.mean(dots):.4f}")
    print(f"  max={np.max(dots):.4f}")
    print(f"  % positive={(dots > 0).mean():.3f}")
    print(dots.shape)
    print(dots)

class TCAVEvaluator:
    def __init__(self, extractor: LLMLayerExtractor):
        self.extractor = extractor

    def score_texts(self, texts: List[str], cav: np.ndarray, layer: int, batch_size=8):
        if len(texts) < 2:
            raise ValueError(f"TCAV requires at least 2 texts, got {len(texts)}.")
        embs = self.extractor.get_hidden_embeddings(texts, layer, batch_size)
        dots = directional_derivatives(embs, cav)
        debug_tcav(dots, f"layer={layer}")
        return {"score": tcav_score(dots), "dots": dots.tolist(), "n_samples": len(dots)}

    def score_pooled_agents(self, agent_outputs: Dict[str, List[str]], cav: np.ndarray,
                            layer: int, batch_size=8, name: str = "all_agents") -> Dict[str, dict]:
        texts = flatten_agent_outputs(agent_outputs)
        result = self.score_texts(texts=texts, cav=cav, layer=layer, batch_size=batch_size)
        return {name: result}

    def tcav_across_layers(self, agent_outputs: Dict[str, List[str]], cav: np.ndarray,
                           layers: List[int], batch_size: int = 8, name: str = "all_agents") -> Dict[int, dict]:
        texts = flatten_agent_outputs(agent_outputs)
        if len(texts) < 2:
            raise ValueError(f"TCAV requires at least 2 texts, got {len(texts)}.")
        results = {}
        for layer in layers:
            embs = self.extractor.get_hidden_embeddings(texts, layer, batch_size)
            dots = directional_derivatives(embs, cav)
            debug_tcav(dots, f"layer={layer}")
            results[layer] = {
                "score": tcav_score(dots),
                "dots": dots.tolist(),
                "mean_dot": float(np.mean(dots)),
                "min_dot": float(np.min(dots)),
                "max_dot": float(np.max(dots)),
                "n_samples": len(dots),
            }
        return {name: results}

# Utils
def flatten_agent_outputs(agent_outputs: Dict[str, List[str]]) -> List[str]:
    texts = []
    for _, outputs in agent_outputs.items():
        for t in outputs:
            if isinstance(t, str) and t.strip():
                texts.append(t.strip())
    return texts

def read_lines(path: str) -> List[str]:
    with open(path, "r") as f:
        return [l.strip() for l in f if l.strip()]

def save_json(path: str, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

# Visualization
def plot_tcav_distribution_layers(tcav_results, save_dir, concept_name):
    os.makedirs(save_dir, exist_ok=True)
    paths = []
    for layer, res in tcav_results.items():
        dots = np.array(res["dots"])
        plt.figure(figsize=(6,4))
        plt.hist(dots, bins=30)
        plt.axvline(0, color="red", linestyle="--")
        plt.xlabel("Directional derivative")
        plt.ylabel("Frequency")
        plt.title(f"TCAV distribution – {concept_name} / layer {layer}")
        plt.tight_layout()
        path = os.path.join(save_dir, f"tcav_dist_{concept_name}_layer{layer}.png")
        plt.savefig(path)
        plt.close()
        paths.append(path)
    return paths

def plot_tcav_scores_layers(tcav_results, save_dir, concept_name):
    os.makedirs(save_dir, exist_ok=True)
    layers = list(tcav_results.keys())
    scores = [tcav_results[l]["score"] for l in layers]
    plt.figure(figsize=(6,4))
    plt.bar([str(l) for l in layers], scores)
    plt.ylim(0, 1)
    plt.xlabel("Layer")
    plt.ylabel("TCAV Score")
    plt.title(f"TCAV Scores Across Layers – {concept_name}")
    plt.tight_layout()
    path = os.path.join(save_dir, f"tcav_scores_{concept_name}.png")
    plt.savefig(path)
    plt.close()
    return path