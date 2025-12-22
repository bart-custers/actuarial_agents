import time
import json
import os
import pandas as pd
import matplotlib.pyplot as plt
from utils.general_utils import make_json_compatible

class WorkflowAudit:
    def __init__(self, log_dir="data/audit"):
        os.makedirs(log_dir, exist_ok=True)
        self.log_dir = log_dir
        self.records = []
        self.start_time = time.time()

    def record_event(self, phase, iteration, action, metadata=None, sent=None, received=None):
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
        }
        self.records.append(entry)

    def finalize(self):
        df = pd.DataFrame(self.records)
        df.to_csv(os.path.join(self.log_dir, "audit_log.csv"), index=False)
        runtime = time.time() - self.start_time
        print(f"\n Total runtime: {runtime:.2f} seconds")
        print(f"Audit log saved to: {self.log_dir}/audit_log.csv")
        return df


import pyagrum as gum
import math

class UncertaintyGraphBN:
    def __init__(self):
        # --------------------------------------------------
        # Define BN structure (success view, AND-gates)
        # --------------------------------------------------
        self.bn = gum.fastBN(
            "DataOK->ModelOK->ReviewOK->ExplainOK->WorkflowOK"
        )

        self.node_map = {
            "dataprep": "DataOK",
            "modelling": "ModelOK",
            "reviewing": "ReviewOK",
            "explanation": "ExplainOK",
        }

        self._init_default_cpts()

    # --------------------------------------------------
    # Default CPTs (neutral priors)
    # --------------------------------------------------
    def _init_default_cpts(self):
        for node in self.bn.nodes():
            var = self.bn.variable(node)
            name = var.name()

            if self.bn.parents(node):
                # Temporary placeholder CPTs
                self.bn.cpt(node).fillWith([0.1, 0.9])
            else:
                # Root priors default to high confidence
                self.bn.cpt(node).fillWith([0.1, 0.9])

    # --------------------------------------------------
    # Aggregate layered uncertainty
    # --------------------------------------------------
    @staticmethod
    def aggregate_layers(layers):
        clean = [u for u in layers if u is not None]
        if not clean:
            return 0.0
        return 1.0 - math.prod(1.0 - u for u in clean)

    # --------------------------------------------------
    # Inject agent uncertainty into CPTs
    # --------------------------------------------------
    def update_from_metadata(self, metadata):
        for agent, node in self.node_map.items():
            layers = [
                v for k, v in metadata.items()
                if k.startswith(f"unc_{agent}_layer")
            ]

            unc = self.aggregate_layers(layers)
            p_ok = 1.0 - unc

            node_id = self.bn.idFromName(node)

            if not self.bn.parents(node_id):
                # Root node → prior
                self.bn.cpt(node_id)[0] = unc
                self.bn.cpt(node_id)[1] = p_ok
            else:
                # Conditional node → Noisy-AND CPT
                parent_id = self.bn.parents(node_id)[0]

                # Parent = 0 → almost sure failure
                self.bn.cpt(node_id)[{'%s'%self.bn.variable(parent_id).name(): 0}] = [0.99, 0.01]

                # Parent = 1 → agent uncertainty applies
                self.bn.cpt(node_id)[{'%s'%self.bn.variable(parent_id).name(): 1}] = [unc, p_ok]

        # WorkflowOK has no own uncertainty; pure AND
        wf_id = self.bn.idFromName("WorkflowOK")
        parent = self.bn.parents(wf_id)[0]
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
