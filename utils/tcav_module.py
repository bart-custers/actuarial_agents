# tcav_module.py
"""
Complete TCAV implementation for LLM-based agent systems.
Includes:
- Hidden-state extractor for LLMWrapper or HF model
- Concept Activation Vector (CAV) training
- Saving & loading CAVs
- TCAV score evaluation
- Multi-agent evaluation
- Automatic layer selection
"""

import os
import json
from typing import List, Dict, Tuple
import numpy as np
import torch

try:
    from sklearn.svm import LinearSVC
except:
    raise ImportError("scikit-learn required for LinearSVC (pip install scikit-learn)")

try:
    from transformers import AutoTokenizer, AutoModel
except:
    AutoTokenizer = None
    AutoModel = None


# ============================================================
# 1) LLM HIDDEN-STATE EXTRACTOR
# ============================================================

class LLMLayerExtractor:
    """
    Adapter to extract hidden states from:
    - The user's LLMWrapper (preferred)
    - Or a raw HuggingFace transformers model (fallback)
    """

    def __init__(self, llm_wrapper=None, model_name=None, device=None):
        self.llm_wrapper = llm_wrapper
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Fallback: use HF model if no wrapper is provided
        self.use_transformers = False
        if llm_wrapper is None and model_name is not None:
            if AutoTokenizer is None:
                raise RuntimeError("transformers is not installed for fallback mode.")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
            self.model.to(self.device)
            self.model.eval()
            self.use_transformers = True

    def _batch(self, items: List[str], batch_size: int):
        for i in range(0, len(items), batch_size):
            yield items[i:i+batch_size]

    def get_hidden_embeddings(self, texts: List[str], layer: int, batch_size: int = 8) -> np.ndarray:
        """
        Returns (N, hidden_dim) array of sentence-level embeddings from a chosen layer.
        """
        # Case 1: preferred method – user’s LLMWrapper
        if self.llm_wrapper is not None:

            # If the wrapper exposes hidden state extraction:
            if hasattr(self.llm_wrapper, "get_hidden_states_for_texts"):
                out = []
                for batch in self._batch(texts, batch_size):
                    arr = self.llm_wrapper.get_hidden_states_for_texts(batch, layer)
                    out.append(arr)
                return np.vstack(out)

            # If it exposes per-layer embeddings
            if hasattr(self.llm_wrapper, "embed_texts_by_layer"):
                out = []
                for batch in self._batch(texts, batch_size):
                    arr = self.llm_wrapper.embed_texts_by_layer(batch, layer)
                    out.append(np.asarray(arr))
                return np.vstack(out)

        # Case 2: fallback — raw transformer model
        if self.use_transformers:
            vecs = []
            with torch.no_grad():
                for batch in self._batch(texts, batch_size):
                    toks = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(self.device)
                    out = self.model(**toks)
                    hidden = out.hidden_states[layer]  # (batch, seq_len, dim)
                    sent = hidden.mean(dim=1).cpu().numpy()
                    vecs.append(sent)
            return np.vstack(vecs)

        raise RuntimeError("No hidden state extraction available.")

# ============================================================
# 2) TRAINING CONCEPT ACTIVATION VECTORS (CAVs)
# ============================================================

def train_cav(concept_embs: np.ndarray, random_embs: np.ndarray,
              C=0.01, max_iter=20000) -> Tuple[np.ndarray, dict]:
    """
    Train linear SVM to separate concept vs random embeddings.
    Returns CAV vector and metadata.
    """
    X = np.vstack([concept_embs, random_embs])
    y = np.hstack([np.ones(len(concept_embs)), np.zeros(len(random_embs))])

    clf = LinearSVC(C=C, max_iter=max_iter)
    clf.fit(X, y)

    cav = clf.coef_.reshape(-1)
    norm = np.linalg.norm(cav)
    if norm > 0:
        cav = cav / norm

    meta = {
        "C": C,
        "max_iter": max_iter,
        "n_samples": len(X)
    }

    return cav, {"clf": clf, "meta": meta}


# ============================================================
# 3) SAVE & LOAD CAV VECTORS
# ============================================================

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


# ============================================================
# 4) TCAV EVALUATION
# ============================================================

def directional_derivatives(embs: np.ndarray, cav: np.ndarray) -> np.ndarray:
    """
    Dot product of embeddings with CAV direction.
    """
    return embs.dot(cav)


def tcav_score(dots: np.ndarray) -> float:
    """
    Fraction of positive directional derivatives.
    """
    return float((dots > 0).mean())


class TCAVEvaluator:
    """
    Core evaluator for multi-agent TCAV scoring.
    """

    def __init__(self, extractor: LLMLayerExtractor):
        self.extractor = extractor

    def score_texts(self, texts: List[str], cav: np.ndarray, layer: int, batch_size=8):
        embs = self.extractor.get_hidden_embeddings(texts, layer, batch_size)
        dots = directional_derivatives(embs, cav)
        return {
            "score": tcav_score(dots),
            "dots": dots.tolist()
        }

    def score_agents(self, agent_outputs: Dict[str, List[str]],
                     cav: np.ndarray, layer: int, batch_size=8) -> Dict[str, dict]:
        out = {}
        for agent, texts in agent_outputs.items():
            out[agent] = self.score_texts(texts, cav, layer, batch_size)
        return out


# ============================================================
# 5) BEST LAYER SELECTION
# ============================================================

def pick_best_layer(extractor: LLMLayerExtractor,
                    concept_texts: List[str],
                    random_texts: List[str],
                    candidate_layers: List[int],
                    batch_size=8) -> int:
    """
    Choose layer with strongest concept activation.
    """
    best_layer = candidate_layers[0]
    best_score = -1.0

    for layer in candidate_layers:
        ce = extractor.get_hidden_embeddings(concept_texts, layer, batch_size)
        re = extractor.get_hidden_embeddings(random_texts, layer, batch_size)
        cav, meta = train_cav(ce, re)
        dots = directional_derivatives(ce, cav)
        score = tcav_score(dots)
        if score > best_score:
            best_score = score
            best_layer = layer

    return best_layer


# ============================================================
# 6) UTILS
# ============================================================

def read_lines(path: str) -> List[str]:
    with open(path, "r") as f:
        return [l.strip() for l in f if l.strip()]


def save_json(path: str, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
