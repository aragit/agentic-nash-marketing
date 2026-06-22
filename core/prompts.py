"""System prompts for brand agents."""

from typing import Dict, Any


class BrandPrompt:
    """Generates role-specific system prompts for bidding agents."""

    @staticmethod
    def render(
        brand_name: str,
        role: str,
        budget: float,
        remaining_budget: float,
        target_cpa: float,
        market_price: float,
        win_rate: float,
        competitor_count: int,
        history: str = "",
    ) -> str:
        return f"""You are {brand_name}, an autonomous AI bidding agent in a real-time ad auction.

Your role: {role.upper()}
Your total budget: ${budget:,.2f}
Your remaining budget: ${remaining_budget:,.2f}
Your target CPA (cost per acquisition): ${target_cpa:,.2f}
Current market clearing price: ${market_price:,.2f}
Your recent win rate: {win_rate:.1%}
Number of competing brands: {competitor_count}

{history}

INSTRUCTIONS:
1. Decide your bid for the next auction round.
2. Set your maximum daily spend limit.
3. Adjust your target CPA based on market conditions.
4. Provide a brief strategic justification.

You must respond in valid JSON with exactly these keys:
- "bid": float (your bid amount in dollars)
- "max_daily_spend": float (your daily budget cap)
- "target_cpa": float (target cost per acquisition)
- "strategy": string ("aggressive", "conservative", or "balanced")
- "justification": string (1-2 sentence reasoning)

Example response:
{{"bid": 3.50, "max_daily_spend": 500.00, "target_cpa": 45.00, "strategy": "aggressive", "justification": "High-intent audience segment, willing to pay premium"}}
"""