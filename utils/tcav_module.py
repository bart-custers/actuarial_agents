import os
import json
from typing import List, Dict, Tuple, Any
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
    def __init__(self, llm_wrapper=None, model_name=None, device=None):
        self.llm_wrapper = llm_wrapper
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def _batch(self, items: List[str], batch_size: int):
        for i in range(0, len(items), batch_size):
            yield items[i:i+batch_size]

    def get_hidden_embeddings(self, texts: List[str], layer: int, batch_size: int = 8) -> np.ndarray:
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

def summarize_dots(dots: np.ndarray) -> Dict[str, float]:
    return {
        "tcav_score": float((dots > 0).mean()),
        "mean_dot": float(np.mean(dots)),
        "median_dot": float(np.median(dots)),
        "std_dot": float(np.std(dots)),
        "min_dot": float(np.min(dots)),
        "max_dot": float(np.max(dots)),
        "n": int(len(dots)),
    }

def random_cavs_like(
    cav: np.ndarray,
    n: int = 20,
    seed: int = 42
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    r = rng.normal(size=(n, cav.shape[0]))
    return r / np.linalg.norm(r, axis=1, keepdims=True)

def tcav_with_random_baseline(
    embs: np.ndarray,
    cav: np.ndarray,
    n_random: int = 20
) -> Dict[str, Any]:

    dots = directional_derivatives(embs, cav)
    main = summarize_dots(dots)

    rand_scores = []
    for rcav in random_cavs_like(cav, n_random):
        rdots = directional_derivatives(embs, rcav)
        rand_scores.append((rdots > 0).mean())

    return {
        "main": main,
        "dots": dots.tolist(),
        "random_tcav_scores": rand_scores,
        "random_mean": float(np.mean(rand_scores)),
        "random_std": float(np.std(rand_scores)),
    }

# TCAV Evaluator
class TCAVEvaluator:
    def __init__(self, extractor: LLMLayerExtractor):
        self.extractor = extractor

    def evaluate_texts(
        self,
        texts: List[str],
        cav: np.ndarray,
        layer: int,
        batch_size: int = 8,
        n_random: int = 20,
    ) -> Dict[str, Any]:

        if len(texts) < 2:
            raise ValueError("TCAV requires at least 2 texts.")

        embs = self.extractor.get_hidden_embeddings(
            texts=texts,
            layer=layer,
            batch_size=batch_size,
        )

        return tcav_with_random_baseline(
            embs=embs,
            cav=cav,
            n_random=n_random,
        )

    def tcav_across_layers(
        self,
        texts: List[str],
        cav: np.ndarray,
        layers: List[int],
        batch_size: int = 8,
        n_random: int = 20,
    ) -> Dict[int, Dict[str, Any]]:

        results = {}
        for layer in layers:
            embs = self.extractor.get_hidden_embeddings(texts, layer, batch_size)
            results[layer] = tcav_with_random_baseline(
                embs=embs,
                cav=cav,
                n_random=n_random,
            )
        return results
    
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
def plot_tcav_across_layers(results: Dict[int, Dict[str, Any]], title: str = ""):
    layers = sorted(results.keys())
    scores = [results[l]["main"]["tcav_score"] for l in layers]

    plt.figure()
    plt.plot(layers, scores, marker="o")
    plt.axhline(0.5, linestyle="--")
    plt.xlabel("Layer")
    plt.ylabel("TCAV score")
    plt.title(title or "TCAV across layers")
    plt.tight_layout()
    store_path = os.path.join("data/evaluation/tcav_plots", "tcav_scores_across_layers.png")
    plt.savefig(store_path)
    plt.close()

def plot_directional_derivatives(dots: np.ndarray, title: str = ""):
    plt.figure()
    plt.hist(dots, bins=40)
    plt.axvline(0.0, linestyle="--")
    plt.xlabel("Directional derivative")
    plt.ylabel("Count")
    plt.title(title or "Directional derivative distribution")
    plt.tight_layout()
    store_path = os.path.join("data/evaluation/tcav_plots", "directional_derivatives.png")
    plt.savefig(store_path)
    plt.close()

def tcav_results_to_dataframe(results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for name, res in results.items():
        row = {"name": name}
        row.update(res["main"])
        row["random_mean"] = res["random_mean"]
        row["random_std"] = res["random_std"]
        rows.append(row)
    return pd.DataFrame(rows)

# Run full TCAV analysis
def run_tcav_analysis(
        llm_wrapper,
        texts: dict,
        concept_file: str = "utils/tcav_concepts.txt",
        random_file: str = "utils/random_concepts.txt",
        layers_to_test: list = [16, 20, 24],
        store_dir: str = "data/evaluation/tcav_plots",
    ) -> str:

    extractor = LLMLayerExtractor(llm_wrapper=llm_wrapper)

    concept_texts = read_lines(concept_file)
    random_texts = read_lines(random_file)

    concept_embs = extractor.get_hidden_embeddings(concept_texts, layer=20)
    random_embs = extractor.get_hidden_embeddings(random_texts, layer=20)

    cav, cav_meta = train_cav(concept_embs, random_embs)

    all_texts = [
        t for outputs in texts.values()
        for t in outputs if isinstance(t, str) and t.strip()
    ]

    evaluator = TCAVEvaluator(extractor)
    tcav_results = evaluator.tcav_across_layers(
        texts=all_texts,
        cav=cav,
        layers=layers_to_test,
        n_random=20,
        batch_size=8,
    )

    os.makedirs(store_dir, exist_ok=True)

    plot_tcav_across_layers(tcav_results, title="TCAV across layers: actuarial reasoning")
    for layer, res in tcav_results.items():
        plot_directional_derivatives(
            dots=res["dots"],
            title=f"Directional derivatives: layer {layer}"
        )

    df = tcav_results_to_dataframe(
        {f"layer_{k}": v for k, v in tcav_results.items()}
    )
    df_path = os.path.join(store_dir, "tcav_summary.csv")
    df.to_csv(df_path, index=False)