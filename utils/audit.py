import time
import os
import pandas as pd
from utils.general_utils import make_json_compatible
import pyagrum as gum
import pyagrum.lib.image as gumimage
import matplotlib.pyplot as plt
from pathlib import Path

class WorkflowAudit:
    def __init__(self, log_dir="data/audit"):
        os.makedirs(log_dir, exist_ok=True)
        self.log_dir = log_dir
        self.records = []
        self.start_time = time.time()

    def record_event(self, phase, iteration, action, metadata=None, sent=None, received=None, uncertainty_posterior=None):
        """Record an audit event. Optionally include the sent message and received response.

        sent / received should be JSON-serializable or will be converted using
        make_json_compatible.
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
        df = pd.DataFrame(self.records)
        df.to_csv(os.path.join(self.log_dir, "audit_log.csv"), index=False)
        runtime = time.time() - self.start_time
        print(f"\n Total runtime: {runtime:.2f} seconds")
        print(f"Audit log saved to: {self.log_dir}/audit_log.csv")
        return df

# class UncertaintyGraphBN:
#     def __init__(self):
#         # --------------------------------------------------
#         # Define BN structure (FINAL schema)
#         # --------------------------------------------------
#         self.bn = gum.fastBN(
#             "DataOK->ReviewDataOK;"
#             "ReviewDataOK->ModelOK;"
#             "ModelOK->WorkflowOK;"
#             "ReviewModelOK->WorkflowOK;"
#             "ExplainOK->WorkflowOK"
#         )

#         self.node_map = {
#             "dataprep": "DataOK",
#             "review_dataprep": "ReviewDataOK",
#             "modelling": "ModelOK",
#             "review_model": "ReviewModelOK",
#             "explanation": "ExplainOK",
#         }

#         self.phase_to_node = {
#             "unc_dataprep": "DataOK",
#             "unc_review_dataprep": "ReviewDataOK",
#             "unc_modelling": "ModelOK",
#             "unc_review_model": "ReviewModelOK",
#             "unc_explanation": "ExplainOK",
#         }

#         self._init_default_cpts()

#     # --------------------------------------------------
#     # Default CPTs (Instantiation-only, Tensor-safe)
#     # --------------------------------------------------
#     def _init_default_cpts(self):

#         # ---------- DataOK (root) ----------
#         cpt = self.bn.cpt("DataOK")
#         inst = gum.Instantiation(cpt)
#         inst.chgVal("DataOK", 0)
#         cpt[inst] = 0.01
#         inst.chgVal("DataOK", 1)
#         cpt[inst] = 0.99

#         # ---------- ReviewDataOK | DataOK ----------
#         cpt = self.bn.cpt("ReviewDataOK")
#         inst = gum.Instantiation(cpt)

#         inst.chgVal("DataOK", 0)
#         inst.chgVal("ReviewDataOK", 0)
#         cpt[inst] = 0.95
#         inst.chgVal("ReviewDataOK", 1)
#         cpt[inst] = 0.05

#         inst.chgVal("DataOK", 1)
#         inst.chgVal("ReviewDataOK", 0)
#         cpt[inst] = 0.05
#         inst.chgVal("ReviewDataOK", 1)
#         cpt[inst] = 0.95

#         # ---------- ModelOK | ReviewDataOK ----------
#         cpt = self.bn.cpt("ModelOK")
#         inst = gum.Instantiation(cpt)

#         inst.chgVal("ReviewDataOK", 0)
#         inst.chgVal("ModelOK", 0)
#         cpt[inst] = 0.90
#         inst.chgVal("ModelOK", 1)
#         cpt[inst] = 0.10

#         inst.chgVal("ReviewDataOK", 1)
#         inst.chgVal("ModelOK", 0)
#         cpt[inst] = 0.10
#         inst.chgVal("ModelOK", 1)
#         cpt[inst] = 0.90

#         # ---------- ReviewModelOK (root) ----------
#         cpt = self.bn.cpt("ReviewModelOK")
#         inst = gum.Instantiation(cpt)
#         inst.chgVal("ReviewModelOK", 0)
#         cpt[inst] = 0.05
#         inst.chgVal("ReviewModelOK", 1)
#         cpt[inst] = 0.95

#         # ---------- ExplainOK (root) ----------
#         cpt = self.bn.cpt("ExplainOK")
#         inst = gum.Instantiation(cpt)
#         inst.chgVal("ExplainOK", 0)
#         cpt[inst] = 0.05
#         inst.chgVal("ExplainOK", 1)
#         cpt[inst] = 0.95

#         # ---------- WorkflowOK = AND(ModelOK, ReviewModelOK, ExplainOK) ----------
#         cpt = self.bn.cpt("WorkflowOK")
#         inst = gum.Instantiation(cpt)

#         inst.setFirst()
#         while not inst.end():
#             ok = (
#                 inst["ModelOK"] == 1
#                 and inst["ReviewModelOK"] == 1
#                 and inst["ExplainOK"] == 1
#             )

#             # Explicitly set both states
#             inst.chgVal("WorkflowOK", 0)
#             cpt[inst] = 0.01 if ok else 0.99

#             inst.chgVal("WorkflowOK", 1)
#             cpt[inst] = 0.99 if ok else 0.01

#             inst.inc()

#     # --------------------------------------------------
#     # Aggregate layered uncertainty
#     # --------------------------------------------------
#     @staticmethod
#     def aggregate_layers(layers):
#         clean = [u for u in layers if u is not None]
#         return sum(clean) / len(clean) if clean else 0.0

#     # --------------------------------------------------
#     # Inject agent uncertainty into CPTs
#     # --------------------------------------------------
#     def update_from_metadata(self, metadata, active_phase=None):

#         phase_to_prefix = {
#             "dataprep": ["unc_dataprep"],
#             "review_dataprep": ["unc_review_dataprep"],
#             "modelling": ["unc_modelling"],
#             "review_model": ["unc_review_model"],
#             "explanation": ["unc_explanation"],
#         }

#         for prefix, node in self.phase_to_node.items():

#             if active_phase and prefix not in phase_to_prefix.get(active_phase, []):
#                 continue

#             layers = [v for k, v in metadata.items() if k.startswith(prefix)]
#             if not layers:
#                 continue

#             unc = self.aggregate_layers(layers)
#             p_ok = 1.0 - unc

#             cpt = self.bn.cpt(node)
#             inst = gum.Instantiation(cpt)
#             parents = list(self.bn.parents(self.bn.idFromName(node)))

#             if not parents:
#                 inst.chgVal(node, 0)
#                 cpt[inst] = unc
#                 inst.chgVal(node, 1)
#                 cpt[inst] = p_ok
#             else:
#                 parent = self.bn.variable(parents[0]).name()

#                 inst.chgVal(parent, 0)
#                 inst.chgVal(node, 0)
#                 cpt[inst] = 0.95
#                 inst.chgVal(node, 1)
#                 cpt[inst] = 0.05

#                 inst.chgVal(parent, 1)
#                 inst.chgVal(node, 0)
#                 cpt[inst] = unc
#                 inst.chgVal(node, 1)
#                 cpt[inst] = p_ok

#         # Re-apply AND-gate (WorkflowOK has no direct uncertainty)
#         cpt = self.bn.cpt("WorkflowOK")
#         inst = gum.Instantiation(cpt)

#         inst.setFirst()
#         while not inst.end():
#             ok = (
#                 inst["ModelOK"] == 1
#                 and inst["ReviewModelOK"] == 1
#                 and inst["ExplainOK"] == 1
#             )

#             # Explicitly set both states
#             inst.chgVal("WorkflowOK", 0)
#             cpt[inst] = 0.01 if ok else 0.99

#             inst.chgVal("WorkflowOK", 1)
#             cpt[inst] = 0.99 if ok else 0.01

#             inst.inc()

#     # --------------------------------------------------
#     # Inference
#     # --------------------------------------------------
#     def infer(self):
#         ie = gum.LazyPropagation(self.bn)
#         ie.makeInference()
#         return {
#             self.bn.variable(n).name(): ie.posterior(n)[1]
#             for n in self.bn.nodes()
#         }

#     # --------------------------------------------------
#     # Debug & visualisation
#     # --------------------------------------------------
#     def debug_print(self):
#         print("\n=== Bayesian Network Structure ===")
#         print(self.bn)
#         print("\n=== CPTs ===")
#         for node in self.bn.nodes():
#             print(self.bn.variable(node).name())
#             print(self.bn.cpt(node))

#     def save_structure(self, filename="bn_structure.png", **kwargs):
#         Path(filename).parent.mkdir(parents=True, exist_ok=True)
#         gumimage.export(self.bn, filename, **kwargs)

#     def save_posteriors(self, filename="bn_with_inference.png", evs=None, targets=None, **kwargs):
#         Path(filename).parent.mkdir(parents=True, exist_ok=True)
#         gumimage.exportInference(
#             self.bn,
#             filename,
#             evs=evs or {},
#             targets=targets or set(),
#             **kwargs,
#         )
from itertools import product

class UncertaintyGraphBN:
    def __init__(self):
        # --------------------------------------------------
        # Define BN structure (success view, AND-gates)
        # --------------------------------------------------
        self.bn = gum.fastBN(
            "DataOK->ReviewDataOK;"
            "ReviewDataOK->ModelOK;"
            "ModelOK->WorkflowOK;"
            "ReviewModelOK->WorkflowOK;"
            "ExplainOK->WorkflowOK"
        )

        self.node_map = {
            "dataprep": "DataOK",
            "review_dataprep": "ReviewDataOK",
            "modelling": "ModelOK",
            "review_model": "ReviewModelOK",
            "explanation": "ExplainOK",
        }

        self.phase_to_node = {
            "unc_dataprep": "DataOK",
            "unc_review_dataprep": "ReviewDataOK",
            "unc_modelling": "ModelOK",
            "unc_review_model": "ReviewModelOK",
            "unc_explanation": "ExplainOK",
        }

        self._init_default_cpts()

    # --------------------------------------------------
    # Default CPTs (neutral priors)
    # --------------------------------------------------
    def _init_default_cpts(self):
        # Root node
        self.bn.cpt("DataOK")[:] = [0.01, 0.99]

        # ReviewDataOK conditional on DataOK
        cpt_rd = self.bn.cpt("ReviewDataOK")
        cpt_rd[{"DataOK": 1}] = [0.05, 0.95]
        cpt_rd[{"DataOK": 0}] = [0.95, 0.05]

        # ModelOK conditional on ReviewDataOK
        cpt_m = self.bn.cpt("ModelOK")
        cpt_m[{"ReviewDataOK": 1}] = [0.10, 0.90]
        cpt_m[{"ReviewDataOK": 0}] = [0.90, 0.10]

        # ReviewModelOK
        cpt_rm = self.bn.cpt("ReviewModelOK")[:] = [0.01, 0.99]
        #cpt_rm[{"ModelOK": 1}] = [0.05, 0.95]
        #cpt_rm[{"ModelOK": 0}] = [0.95, 0.05]

        # ExplainOK
        cpt_e = self.bn.cpt("ExplainOK")[:] = [0.01, 0.99]
        #cpt_e[{"ReviewModelOK": 1}] = [0.05, 0.95]
        #cpt_e[{"ReviewModelOK": 0}] = [0.95, 0.05]

        # WorkflowOK conditional on ModelOK, ReviewModelOK, ExplainOK
        cpt_wf = self.bn.cpt("WorkflowOK")

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
        clean = [u for u in layers if u is not None]
        if not clean:
            return 0.0
        return sum(clean) / len(clean)

    # --------------------------------------------------
    # Inject agent uncertainty into CPTs
    # --------------------------------------------------
    def update_from_metadata(self, metadata, active_phase=None):
        phase_to_prefix = {
            "dataprep": ["unc_dataprep"],
            "review_dataprep": ["unc_review_dataprep"],
            "modelling": ["unc_modelling"],
            "review_model": ["unc_review_model"],
            "explanation": ["unc_explanation"],
        }

        for prefix, node in self.phase_to_node.items():
            if active_phase:
                allowed_prefixes = phase_to_prefix.get(active_phase, [])
                if prefix not in allowed_prefixes:
                    continue

            layers = [v for k, v in metadata.items() if k.startswith(prefix)]
            if not layers:
                continue

            unc = self.aggregate_layers(layers)   # uncertainty from metadata
            p_ok = 1 - unc
            node_id = self.bn.idFromName(node)
            parents = [self.bn.variable(p).name() for p in self.bn.parents(node_id)]

            if not parents:
                # Root node → prior
                self.bn.cpt(node_id)[:] = [unc, p_ok]
            else:
                # Multi-parent node → Noisy-AND style update
                for state in product([0, 1], repeat=len(parents)):
                    state_dict = dict(zip(parents, state))
                    self.bn.cpt(node_id)[state_dict] = [unc, p_ok]

    # --------------------------------------------------
    # Inference
    # --------------------------------------------------
    def infer(self):
        ie = gum.LazyPropagation(self.bn)
        ie.makeInference()
        results = {}
        for node in self.bn.nodes():
            name = self.bn.variable(node).name()
            results[name] = ie.posterior(node)[1]  # P(node = OK)
        return results
    
    def debug_print(self):
        print("\n=== Bayesian Network Structure ===")
        print(self.bn)

        print("\n=== CPTs ===")
        for node in self.bn.nodes():
            var = self.bn.variable(node)
            print(f"\nNode: {var.name()}  |  States: {var.labels()}")
            print(self.bn.cpt(node))
    
    def save_structure(self, filename="bn_structure.png", **kwargs):
        """
        Save the Bayesian network structure as an image file.
        """
        # Ensure the directory exists
        Path(filename).parent.mkdir(parents=True, exist_ok=True)

        # Export the BN
        gumimage.export(self.bn, filename, **kwargs)
    
    def save_posteriors(self, filename="bn_with_inference.png", evs=None, targets=None, **kwargs):
        """
        Save a graphical representation of the inference (posteriors) as an image.
        """
        # Ensure the directory exists
        Path(filename).parent.mkdir(parents=True, exist_ok=True)

        # Export inference
        gumimage.exportInference(
            self.bn,
            filename,
            evs=evs or {},        # evidence if any
            targets=targets or set(),  # nodes to show (empty=set => all)
            **kwargs
        )

