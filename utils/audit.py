import time
import os
import json
import pandas as pd
from utils.general_utils import make_json_compatible, save_json_safe
import pyagrum as gum
import pyagrum.lib.image as gumimage
from itertools import product
from pathlib import Path

class WorkflowAudit:
    """
    Audit logger for workflow execution events.

    Records each action or phase with timestamps, optional messages, responses,
    and uncertainty estimates. Can finalize and export the log as a CSV.
    """
    def __init__(self, log_dir="data/audit"):
        """
        Initialize the audit logger.

        Args:
            log_dir (str): Directory to store audit logs. Will be created if missing.
        """
        os.makedirs(log_dir, exist_ok=True)
        self.log_dir = log_dir
        self.records = []
        self.start_time = time.time() # Track workflow runtime

    def record_event(self, phase, iteration, action, metadata=None, sent=None, received=None, uncertainty_posterior=None):
        """
        Record a workflow event.

        Args:
            phase (str): Current phase of the workflow.
            iteration (int): Iteration number (if applicable).
            action (str): Description of the action performed.
            metadata (dict, optional): Additional metadata, e.g., status info.
            sent (any, optional): Message/data sent (will be JSON-compatible).
            received (any, optional): Response received (will be JSON-compatible).
            uncertainty_posterior (any, optional): Optional uncertainty estimate.

        Notes:
            sent, received, and uncertainty_posterior are converted to
            JSON-serializable formats if needed.
        """
        entry = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "phase": phase,
            "iteration": iteration,
            "action": action,
            "status": metadata.get("status") if metadata else None,
            "sent": make_json_compatible(sent) if sent is not None else None,
            "received": make_json_compatible(received) if received is not None else None,
            "uncertainty_posterior": make_json_compatible(uncertainty_posterior) if uncertainty_posterior is not None else None
        }
        self.records.append(entry)

    def finalize(self):
        """
        Save audit log to CSV and report total runtime.

        Returns:
            pd.DataFrame: DataFrame containing all logged events.
        """
        df = pd.DataFrame(self.records)
        df.to_csv(os.path.join(self.log_dir, "audit_log.csv"), index=False)
        runtime = time.time() - self.start_time
        print(f"\n Total runtime: {runtime:.2f} seconds")
        print(f"Audit log saved to: {self.log_dir}/audit_log.csv")
        return df


class UncertaintyGraphBN:
    """
    Bayesian network representing workflow uncertainty.

    Each workflow phase (data prep, modeling, review, explanation) is a node. The network
    allows propagating uncertainties from individual agents or steps to an overall
    workflow-level uncertainty.

    Two structurally-identical pyagrum BayesNets are kept (`bn_lo`, `bn_hi`) rather than one,
    so that a node calibrated from HIP-LLM's operational-profile failure-probability estimate
    (see docs/uncertainty_operational_profile.md) can carry its lower/upper bound all the way
    through inference as an interval, instead of collapsing to a single point estimate —
    mirroring the optimistic/pessimistic BN pattern demonstrated in
    smiles-safety-agent-parrallel-test.ipynb (cell 25). Nodes that have not been calibrated
    (still driven by `update_from_metadata`'s self-reported/logprob scalar) simply get the
    same CPT in both variants, so `infer()` returns the same value twice for them.
    """
    def __init__(self):
        # --------------------------------------------------
        # Define BN structure (success view, AND-gates) -- built twice, identically
        # --------------------------------------------------
        self.bn_lo = self._build_structure()   # HIP-LLM lower failure bound -> optimistic
        self.bn_hi = self._build_structure()   # HIP-LLM upper failure bound -> pessimistic

        # Mapping workflow phases to BN nodes
        self.node_map = {
            "dataprep": "DataOK",
            "review_dataprep": "ReviewDataOK",
            "modelling": "ModelOK",
            "review_model": "ReviewModelOK",
            "explanation": "ExplainOK",
        }

        # Prefix mapping for uncertainty metadata
        self.phase_to_node = {
            "unc_dataprep": "DataOK",
            "unc_review_dataprep": "ReviewDataOK",
            "unc_modelling": "ModelOK",
            "unc_review_model": "ReviewModelOK",
            "unc_explanation": "ExplainOK",
        }

        # Nodes whose CPT is driven by a HIP-LLM calibration rather than by
        # update_from_metadata's live self-reported scalar (see calibrate_node/load_calibration).
        self.calibrated_nodes = set()
        self._calibration_values = {}

        self._init_default_cpts()

    @staticmethod
    def _build_structure():
        return gum.fastBN(
            "DataOK->ReviewDataOK;"
            "ReviewDataOK->ModelOK;"
            "ModelOK->WorkflowOK;"
            "ReviewModelOK->WorkflowOK;"
            "ExplainOK->WorkflowOK"
        )

    # --------------------------------------------------
    # Default CPTs (neutral priors) -- identical in both variants until calibrated
    # --------------------------------------------------
    def _init_default_cpts(self):
        """
        Initialize default Conditional Probability Tables (CPTs) with neutral priors.
        """
        for bn in (self.bn_lo, self.bn_hi):
            # Root node
            bn.cpt("DataOK")[:] = [0.01, 0.99]

            # ReviewDataOK conditional on DataOK
            cpt_rd = bn.cpt("ReviewDataOK")
            cpt_rd[{"DataOK": 1}] = [0.05, 0.95]
            cpt_rd[{"DataOK": 0}] = [0.95, 0.05]

            # ModelOK conditional on ReviewDataOK
            cpt_m = bn.cpt("ModelOK")
            cpt_m[{"ReviewDataOK": 1}] = [0.10, 0.90]
            cpt_m[{"ReviewDataOK": 0}] = [0.90, 0.10]

            # ReviewModelOK
            bn.cpt("ReviewModelOK")[:] = [0.01, 0.99]

            # ExplainOK
            bn.cpt("ExplainOK")[:] = [0.01, 0.99]

            # WorkflowOK conditional on ModelOK, ReviewModelOK, ExplainOK
            cpt_wf = bn.cpt("WorkflowOK")

            for model_ok, review_model_ok, explain_ok in product([0, 1], repeat=3):
                if model_ok == 1 and review_model_ok == 1 and explain_ok == 1:
                    # All parents OK → workflow OK with high probability
                    cpt_wf[{
                        "ModelOK": model_ok,
                        "ReviewModelOK": review_model_ok,
                        "ExplainOK": explain_ok
                    }] = [0.01, 0.99]
                else:
                    # Any parent fails → workflow almost surely fails
                    cpt_wf[{
                        "ModelOK": model_ok,
                        "ReviewModelOK": review_model_ok,
                        "ExplainOK": explain_ok
                    }] = [0.99, 0.01]

    # --------------------------------------------------
    # Aggregate layered uncertainty
    # --------------------------------------------------
    @staticmethod
    def aggregate_layers(layers):
        """
        Compute average uncertainty across multiple layers.

        Args:
            layers (list[float]): List of uncertainty values (0-1).

        Returns:
            float: Mean uncertainty; 0 if no valid layers.
        """
        clean = [u for u in layers if u is not None]
        if not clean:
            return 0.0
        return sum(clean) / len(clean)

    # --------------------------------------------------
    # Shared CPT-row writer (same value across all parent-state combinations, matching the
    # pre-existing simplification in update_from_metadata -- see
    # docs/uncertainty_operational_profile.md, "known limitation carried over").
    # --------------------------------------------------
    @staticmethod
    def _set_node_cpt(bn, node, p_ok):
        node_id = bn.idFromName(node)
        parents = [bn.variable(p).name() for p in bn.parents(node_id)]
        if not parents:
            bn.cpt(node_id)[:] = [1 - p_ok, p_ok]
        else:
            for state in product([0, 1], repeat=len(parents)):
                state_dict = dict(zip(parents, state))
                bn.cpt(node_id)[state_dict] = [1 - p_ok, p_ok]

    # --------------------------------------------------
    # Inject agent uncertainty into CPTs (nodes without a HIP-LLM calibration only)
    # --------------------------------------------------
    def update_from_metadata(self, metadata, active_phase=None):
        """
        Update BN CPTs using agent-reported uncertainties.

        Args:
            metadata (dict): Mapping of phase uncertainty keys to values.
            active_phase (str, optional): Only update nodes for this phase.

        Notes:
            Nodes already present in `self.calibrated_nodes` (see calibrate_node/
            load_calibration) are skipped -- their CPT is driven by HIP-LLM's operational-
            profile calibration instead, and would otherwise be overwritten every call.
        """
        phase_to_prefix = {
            "dataprep": ["unc_dataprep"],
            "review_dataprep": ["unc_review_dataprep"],
            "modelling": ["unc_modelling"],
            "review_model": ["unc_review_model"],
            "explanation": ["unc_explanation"],
        }

        for prefix, node in self.phase_to_node.items():
            if node in self.calibrated_nodes:
                continue

            if active_phase:
                allowed_prefixes = phase_to_prefix.get(active_phase, [])
                if prefix not in allowed_prefixes:
                    continue

            layers = [v for k, v in metadata.items() if k.startswith(prefix)]
            if not layers:
                continue

            unc = self.aggregate_layers(layers)   # uncertainty from metadata
            p_ok = 1 - unc
            # Uncalibrated nodes carry no interval of their own -- write the same scalar into
            # both variants; the interval in the final posterior comes only from calibrated nodes.
            self._set_node_cpt(self.bn_lo, node, p_ok)
            self._set_node_cpt(self.bn_hi, node, p_ok)

    # --------------------------------------------------
    # HIP-LLM operational-profile calibration
    # --------------------------------------------------
    def calibrate_node(self, node, outcomes, strata, profile=None, settings=None):
        """
        Fit HIP-LLM's OperationalFailureProb on historical (outcome, stratum) labels for
        `node` and write the resulting failure-probability bounds into this node's CPT --
        the lower bound into `bn_lo` (optimistic), the upper bound into `bn_hi` (pessimistic).

        See utils/mine_hip_labels.py for how `outcomes`/`strata` are mined from
        data/memory/central_memory.json, and docs/uncertainty_operational_profile.md
        (§4.4-4.5) for the design this implements.

        Args:
            node (str): BN node name, e.g. "DataOK".
            outcomes (list[int]): 0/1 historical outcome labels.
            strata (list[str]): matching stratum label per outcome.
            profile (dict[str, float], optional): operational-profile stratum weights.
                Defaults to the empirical stratum frequency in `strata` itself (see
                docs/uncertainty_operational_profile.md, "Operational profile weights") --
                a placeholder until a real traffic-mix profile is defined.
            settings: passed through to HIPLLM.quick_inference_settings if given; otherwise
                a default of samples=1500, configurations=48 is used (matches the notebook
                precedent).

        Returns:
            dict: {"failure_probability_lower": float, "failure_probability_upper": float}.

        HIP-LLM is imported lazily here (not at module load time) so the rest of this class
        keeps working without the package installed unless calibration is actually requested.
        """
        from HIPLLM import OperationalFailureProb, quick_inference_settings

        if profile is None:
            profile = pd.Series(strata).value_counts(normalize=True).to_dict()

        estimator = OperationalFailureProb(
            profile=profile,
            settings=settings or quick_inference_settings(samples=1500, configurations=48),
        )
        result = estimator.fit(outcomes=outcomes, strata=strata)
        summary = result.summary()
        fail_lo = summary["posterior_expected_failure_lower"]
        fail_hi = summary["posterior_expected_failure_upper"]

        p_ok_lo, p_ok_hi = 1 - fail_lo, 1 - fail_hi
        self._set_node_cpt(self.bn_lo, node, p_ok_lo)
        self._set_node_cpt(self.bn_hi, node, p_ok_hi)
        self.calibrated_nodes.add(node)
        self._calibration_values[node] = {"p_ok_lo": p_ok_lo, "p_ok_hi": p_ok_hi}
        return {"failure_probability_lower": fail_lo, "failure_probability_upper": fail_hi}

    def save_calibration(self, path="data/audit/bn_calibration.json"):
        """
        Persist calibrated nodes' failure-probability bounds so CentralHub can load them at
        construction time without re-fitting HIP-LLM on every workflow run (recalibration is
        a periodic/offline step -- see utils/recalibrate_bn.py).
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        save_json_safe(self._calibration_values, path)
        return path

    def load_calibration(self, path="data/audit/bn_calibration.json"):
        """
        Load a previously-saved calibration (see save_calibration / utils/recalibrate_bn.py).

        Returns:
            bool: True if a calibration file was found and loaded, False if `path` doesn't
            exist yet -- callers should treat False as "fall back to the neutral default
            priors", not as an error.
        """
        if not os.path.exists(path):
            return False
        with open(path, "r") as f:
            payload = json.load(f)
        for node, bounds in payload.items():
            self._set_node_cpt(self.bn_lo, node, bounds["p_ok_lo"])
            self._set_node_cpt(self.bn_hi, node, bounds["p_ok_hi"])
            self.calibrated_nodes.add(node)
            self._calibration_values[node] = bounds
        return True

    # --------------------------------------------------
    # Inference
    # --------------------------------------------------
    def infer(self):
        """
        Perform probabilistic inference on both BN variants.

        Returns:
            dict: {node_name: (p_ok_lower, p_ok_upper)}. For HIP-LLM-calibrated nodes this is
            a real interval; for nodes still driven by update_from_metadata, both entries are
            equal (no interval information exists for them yet).
        """
        results = {}
        for key, bn in (("lo", self.bn_lo), ("hi", self.bn_hi)):
            ie = gum.LazyPropagation(bn)
            ie.makeInference()
            for node in bn.nodes():
                name = bn.variable(node).name()
                results.setdefault(name, {})[key] = ie.posterior(node)[1]  # P(node = OK)
        return {name: (bounds["lo"], bounds["hi"]) for name, bounds in results.items()}

    def debug_print(self):
        for label, bn in (("OPTIMISTIC (bn_lo)", self.bn_lo), ("PESSIMISTIC (bn_hi)", self.bn_hi)):
            print(f"\n=== Bayesian Network Structure [{label}] ===")
            print(bn)

            print("\n=== CPTs ===")
            for node in bn.nodes():
                var = bn.variable(node)
                print(f"\nNode: {var.name()}  |  States: {var.labels()}")
                print(bn.cpt(node))

    def save_structure(self, filename="bn_structure.png", **kwargs):
        """
        Save both BN variants' structure as image files (suffixed `_optimistic`/`_pessimistic`).
        """
        base = Path(filename)
        base.parent.mkdir(parents=True, exist_ok=True)

        gumimage.export(self.bn_lo, str(base.with_name(f"{base.stem}_optimistic{base.suffix}")), **kwargs)
        gumimage.export(self.bn_hi, str(base.with_name(f"{base.stem}_pessimistic{base.suffix}")), **kwargs)

    def save_posteriors(self, filename="bn_with_inference.png", evs=None, targets=None, **kwargs):
        """
        Save both BN variants' inference (posteriors) as image files (suffixed
        `_optimistic`/`_pessimistic`).
        """
        base = Path(filename)
        base.parent.mkdir(parents=True, exist_ok=True)

        for suffix, bn in (("_optimistic", self.bn_lo), ("_pessimistic", self.bn_hi)):
            gumimage.exportInference(
                bn,
                str(base.with_name(f"{base.stem}{suffix}{base.suffix}")),
                evs=evs or {},
                targets=targets or set(),
                **kwargs
            )

