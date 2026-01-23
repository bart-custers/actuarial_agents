DECISION_MAP_REVIEW = {
    "APPROVE": "approve",
    "APPROVE_WITH_NOTES": "approve_with_notes",
    "REQUEST_RECLEAN": "request_reclean",
    "REQUEST_RETRAIN": "request_retrain",
    "ABORT": "abort",
}

ROUTING_MAP_REVIEW = {
    "approve": "proceed",
    "approve_with_notes": "proceed",
    "request_reclean": "reclean_data",
    "request_retrain": "retrain_model",
    "abort": "abort_workflow"
}

DECISION_MAP_EXPLANATION = {
    "APPROVE": "approve",
    "MINOR_ISSUES": "minor_issues",
    "REQUEST_RECLEAN": "request_reclean",
    "REQUEST_RETRAIN": "request_retrain",
    "ABORT": "abort",
}

ROUTING_MAP_EXPLANATION = {
    "approve": "finalize",
    "minor_issues": "consult_actuary",
    "request_reclean": "reclean_data",
    "request_retrain": "retrain_model",
    "abort": "abort_workflow"
}