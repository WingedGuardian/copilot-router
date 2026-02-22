"""Route decision data model.

V1 heuristic classify() has been removed — routing is now plan-based
(see router.py V2). Only the RouteDecision dataclass remains, as it's
used throughout the routing system.
"""

from dataclasses import dataclass


@dataclass
class RouteDecision:
    """Result of routing classification."""

    target: str  # "local", "default", "plan", "escalation", "cloud"
    reason: str  # Why this route was chosen
    model: str   # Specific model identifier to use
