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


class UncertaintyGraphBN:
    def __init__(self):
        # --------------------------------------------------
        # Define BN structure (success view, AND-gates)
        # --------------------------------------------------
        self.bn = gum.fastBN(
            "DataOK"
            "->ReviewDataOK"
            "->ModelOK"
            "->ReviewModelOK"
            "->ExplainOK"
            "->WorkflowOK"
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

        # ReviewModelOK conditional on ModelOK
        cpt_rm = self.bn.cpt("ReviewModelOK")
        cpt_rm[{"ModelOK": 1}] = [0.05, 0.95]
        cpt_rm[{"ModelOK": 0}] = [0.95, 0.05]

        # ExplainOK conditional on ReviewModelOK
        cpt_e = self.bn.cpt("ExplainOK")
        cpt_e[{"ReviewModelOK": 1}] = [0.05, 0.95]
        cpt_e[{"ReviewModelOK": 0}] = [0.95, 0.05]

        # WorkflowOK conditional on ExplainOK
        cpt_wf = self.bn.cpt("WorkflowOK")
        cpt_wf[{"ExplainOK": 1}] = [0.01, 0.99]
        cpt_wf[{"ExplainOK": 0}] = [0.99, 0.01]

    # --------------------------------------------------
    # Aggregate layered uncertainty
    # --------------------------------------------------
    # @staticmethod
    # def aggregate_layers(layers):
    #     clean = [u for u in layers if u is not None]
    #     if not clean:
    #         return 0.0
    #     return 1.0 - math.prod(1.0 - u for u in clean)
    
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
        # phase_to_prefix = {
        #     "DataOK": "unc_dataprep",
        #     "ReviewDataOK": "unc_review_dataprep",
        #     "ModelOK": "unc_modelling",
        #     "ReviewModelOK": "unc_review_model",
        #     "ExplainOK": "unc_explanation",
        # }

        phase_to_prefix = {
            "dataprep": ["unc_dataprep"],
            "review_dataprep": ["unc_review_dataprep"],
            "modelling": ["unc_modelling"],
            "review_model": ["unc_review_model"],
            "explanation": ["unc_explanation"],
        }
        
        for prefix, node in self.phase_to_node.items():

            # Skip if not active
            if active_phase:
                allowed_prefixes = phase_to_prefix.get(active_phase, [])
                if prefix not in allowed_prefixes:
                    continue

            layers = [
                v for k, v in metadata.items()
                if k.startswith(prefix)
            ]

            if not layers:
                continue

            unc = self.aggregate_layers(layers)
            p_ok = 1 - unc

        # for agent, node in self.node_map.items():
        #     layers = [
        #         v for k, v in metadata.items()
        #         if k.startswith(f"unc_{agent}_layer")
        #     ]

        #     unc = self.aggregate_layers(layers)
        #     p_ok = 1.0 - unc

            node_id = self.bn.idFromName(node)

            if not self.bn.parents(node_id):
                # Root node → prior
                self.bn.cpt(node_id)[0] = unc
                self.bn.cpt(node_id)[1] = p_ok
            else:
                # Conditional node → Noisy-AND CPT
                parent_id = list(self.bn.parents(node_id))[0]

                # Parent = 0 → almost sure failure
                self.bn.cpt(node_id)[{'%s'%self.bn.variable(parent_id).name(): 0}] = [0.99, 0.01]

                # Parent = 1 → agent uncertainty applies
                self.bn.cpt(node_id)[{'%s'%self.bn.variable(parent_id).name(): 1}] = [unc, p_ok]

        # WorkflowOK has no own uncertainty; pure AND
        wf_id = self.bn.idFromName("WorkflowOK")
        parent = list(self.bn.parents(wf_id))[0]
        self.bn.cpt(wf_id)[{'ExplainOK': 0}] = [0.99, 0.01]
        self.bn.cpt(wf_id)[{'ExplainOK': 1}] = [0.01, 0.99]

    # --------------------------------------------------
    # Inference (this is the propagation)
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

    def save_node_posterior(self, node_name, filename="posterior_node.png"):
        ie = gum.LazyPropagation(self.bn)
        ie.makeInference()

        posterior = ie.posterior(node_name)
        values = posterior.toarray()
        labels = posterior.variable().labels()

        plt.figure(figsize=(6, 4))
        plt.bar(labels, values)
        plt.title(f"Posterior for {node_name}")
        plt.ylabel("Probability")
        plt.tight_layout()
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(filename, dpi=300)
        plt.close()

