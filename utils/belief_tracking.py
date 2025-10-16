from difflib import SequenceMatcher

def belief_change_tracking(explanation_texts):
    """
    Compute belief-change scores between iterations (simple text similarity metric).
    """
    changes = []
    for i in range(1, len(explanation_texts)):
        prev, curr = explanation_texts[i - 1], explanation_texts[i]
        similarity = SequenceMatcher(None, prev, curr).ratio()
        changes.append(1 - similarity)  # 0 = stable, 1 = total change
    return changes
