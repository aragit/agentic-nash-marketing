"""Budget guardrails to prevent catastrophic depletion."""

import logging
from typing import Dict, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class GuardrailAction:
    """Action taken by the guardrail system."""
    agent_name: str
    original_bid: float
    adjusted_bid: float
    action: str  # "allow", "cap", "block", "emergency"
    reason: str


class BudgetGuardrail:
    """
    Prevents budget depletion through multi-layer guardrails:
    1. Soft cap: Warn when budget < 20%
    2. Hard cap: Block bids when budget < 10%
    3. Emergency: Force conservative strategy when budget < 5%
    """

    SOFT_THRESHOLD = 0.20
    HARD_THRESHOLD = 0.10
    EMERGENCY_THRESHOLD = 0.05

    def check(self, agent_name: str, bid: float, remaining: float, total: float) -> GuardrailAction:
        """Check bid against guardrails and return adjusted action."""
        ratio = remaining / total if total > 0 else 0.0

        if ratio <= self.EMERGENCY_THRESHOLD:
            # Emergency: force minimum viable bid
            adjusted = min(bid, total * 0.01)  # 1% of total
            return GuardrailAction(
                agent_name=agent_name,
                original_bid=bid,
                adjusted_bid=round(adjusted, 2),
                action="emergency",
                reason=f"EMERGENCY: Budget at {ratio:.1%}. Forced minimum bid.",
            )

        if ratio <= self.HARD_THRESHOLD:
            # Hard cap: maximum 5% of remaining
            adjusted = min(bid, remaining * 0.05)
            return GuardrailAction(
                agent_name=agent_name,
                original_bid=bid,
                adjusted_bid=round(adjusted, 2),
                action="cap",
                reason=f"HARD CAP: Budget at {ratio:.1%}. Bid capped at 5% of remaining.",
            )

        if ratio <= self.SOFT_THRESHOLD:
            # Soft cap: warn, allow but log
            logger.warning(f"[{agent_name}] SOFT WARNING: Budget at {ratio:.1%}")
            return GuardrailAction(
                agent_name=agent_name,
                original_bid=bid,
                adjusted_bid=round(bid, 2),
                action="allow",
                reason=f"SOFT WARNING: Budget at {ratio:.1%}. Bid allowed but monitored.",
            )

        # Normal operation
        return GuardrailAction(
            agent_name=agent_name,
            original_bid=bid,
            adjusted_bid=round(bid, 2),
            action="allow",
            reason="Budget healthy. Bid approved.",
        )

    def get_system_status(self, agent_budgets: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """Get overall guardrail system status."""
        statuses = {}
        for name, data in agent_budgets.items():
            remaining = data.get("remaining", 0)
            total = data.get("total", 1)
            ratio = remaining / total
            if ratio <= self.EMERGENCY_THRESHOLD:
                status = "CRITICAL"
            elif ratio <= self.HARD_THRESHOLD:
                status = "WARNING"
            elif ratio <= self.SOFT_THRESHOLD:
                status = "CAUTION"
            else:
                status = "HEALTHY"
            statuses[name] = {
                "status": status,
                "remaining_ratio": round(ratio, 3),
                "remaining_amount": remaining,
            }
        return statuses