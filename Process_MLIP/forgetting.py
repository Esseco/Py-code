"""Summarize a round's relaxation-collapse evidence for reliability review."""


def assess_forgetting(structure_ids, collapse_ids, *, coverage_complete):
    """Count distinct relaxed structures; incomplete coverage cannot clear a round."""
    if type(coverage_complete) is not bool:
        raise ValueError("coverage_complete must be a boolean")
    structures = list(structure_ids)
    collapsed = list(collapse_ids)
    if any(not isinstance(item, str) or not item for item in structures + collapsed):
        raise ValueError("structure identifiers must be nonempty strings")
    structures, collapsed = set(structures), set(collapsed)
    if not structures or not collapsed <= structures:
        raise ValueError("collapse evidence must belong to a nonempty calculation")
    count, total = len(collapsed), len(structures)
    triggered = count * 10 >= total
    return {"structure_count": total, "collapse_count": count,
            "collapse_fraction": count / total,
            "coverage_complete": coverage_complete,
            "reliability_review_required": triggered or not coverage_complete,
            "direct_sampling_required": triggered if coverage_complete else None}
