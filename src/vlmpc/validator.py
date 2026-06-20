"""VLMPC validator — pick the minimum-cost trajectory (SPEC section 8.6)."""

from src.models import CostBreakdown


def select_best(costs: list[CostBreakdown]) -> int:
    """Return the id of the trajectory with the lowest total cost."""
    if not costs:
        raise ValueError("select_best requires at least one cost")
    return min(costs, key=lambda c: c.total_cost).trajectory_id
