"""Nash equilibrium solver for multi-agent competitive bidding."""

import logging
import numpy as np
from typing import Dict, List, Tuple
from scipy.optimize import minimize, LinearConstraint

logger = logging.getLogger(__name__)


class NashEquilibriumSolver:
    """
    Computes mixed-strategy Nash equilibrium for N-player ad auction game.
    
    Each player (brand) chooses a mixed strategy over discrete bid levels.
    The equilibrium is where no player can improve their expected utility
    by unilaterally changing their strategy.
    """

    def __init__(self, bid_levels: List[float] = None):
        self.bid_levels = bid_levels or [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

    def compute_equilibrium(
        self,
        agent_budgets: Dict[str, float],
        agent_valuations: Dict[str, float],  # Value per conversion
        impression_supply: int,
    ) -> Dict[str, any]:
        """
        Compute approximate mixed-strategy Nash equilibrium.
        
        Uses iterative best-response with softmax smoothing.
        """
        n_agents = len(agent_budgets)
        n_levels = len(self.bid_levels)

        if n_agents == 0:
            return {"strategies": {}, "clearing_price": 0.0, "convergence": 0.0}

        # Initialize uniform mixed strategies
        strategies = {
            name: np.ones(n_levels) / n_levels
            for name in agent_budgets.keys()
        }

        # Iterative best response
        max_iterations = 100
        tolerance = 1e-4

        for iteration in range(max_iterations):
            new_strategies = {}

            for agent_name in agent_budgets.keys():
                opponent_names = [n for n in agent_budgets.keys() if n != agent_name]

                # Compute expected utility for each bid level
                utilities = np.zeros(n_levels)

                for i, bid in enumerate(self.bid_levels):
                    expected_utility = self._expected_utility(
                        agent_name=agent_name,
                        bid=bid,
                        agent_budgets=agent_budgets,
                        agent_valuations=agent_valuations,
                        opponent_strategies={
                            n: strategies[n] for n in opponent_names
                        },
                        impression_supply=impression_supply,
                    )
                    utilities[i] = expected_utility

                # Softmax best response (temperature for convergence stability)
                temperature = max(0.1, 1.0 - iteration / max_iterations)
                exp_utils = np.exp(utilities / temperature)
                new_strategies[agent_name] = exp_utils / np.sum(exp_utils)

            # Check convergence
            max_diff = max(
                np.max(np.abs(new_strategies[name] - strategies[name]))
                for name in strategies.keys()
            )

            strategies = new_strategies

            if max_diff < tolerance:
                logger.info(f"Nash equilibrium converged in {iteration + 1} iterations")
                break

        # Compute equilibrium clearing price
        eq_clearing_price = self._equilibrium_clearing_price(
            strategies, agent_budgets, impression_supply
        )

        return {
            "strategies": {
                name: {
                    "distribution": strategies[name].tolist(),
                    "expected_bid": float(np.dot(strategies[name], self.bid_levels)),
                    "bid_levels": self.bid_levels,
                }
                for name in strategies.keys()
            },
            "clearing_price": round(eq_clearing_price, 2),
            "convergence": float(max_diff),
            "iterations": iteration + 1,
        }

    def _expected_utility(
        self,
        agent_name: str,
        bid: float,
        agent_budgets: Dict[str, float],
        agent_valuations: Dict[str, float],
        opponent_strategies: Dict[str, np.ndarray],
        impression_supply: int,
    ) -> float:
        """Compute expected utility for a single bid level."""
        valuation = agent_valuations.get(agent_name, 50.0)
        budget = agent_budgets.get(agent_name, 1000.0)

        # Probability of winning with this bid against opponent mixed strategies
        win_prob = self._win_probability(bid, opponent_strategies, impression_supply)

        # Expected profit = win_prob * (valuation - bid) - but constrained by budget
        if bid > budget * 0.2:  # Can't spend more than 20% per bid
            return -1000.0  # Heavily penalize over-budget bids

        expected_profit = win_prob * (valuation - bid)
        return expected_profit

    def _win_probability(
        self,
        bid: float,
        opponent_strategies: Dict[str, np.ndarray],
        impression_supply: int,
    ) -> float:
        """Probability that this bid wins given opponent mixed strategies."""
        if not opponent_strategies:
            return 1.0

        # Monte Carlo estimate: sample opponent bids
        n_samples = 1000
        wins = 0

        for _ in range(n_samples):
            opponent_bids = []
            for name, strategy in opponent_strategies.items():
                sampled_bid = np.random.choice(self.bid_levels, p=strategy)
                opponent_bids.append(sampled_bid)

            # Count how many opponents bid higher
            higher_bids = sum(1 for b in opponent_bids if b > bid)
            if higher_bids < impression_supply:
                wins += 1

        return wins / n_samples

    def _equilibrium_clearing_price(
        self,
        strategies: Dict[str, np.ndarray],
        agent_budgets: Dict[str, float],
        impression_supply: int,
    ) -> float:
        """Estimate equilibrium clearing price from mixed strategies."""
        # Expected bids from all agents
        expected_bids = []
        for name, strategy in strategies.items():
            expected_bid = np.dot(strategy, self.bid_levels)
            expected_bids.append(expected_bid)

        sorted_bids = sorted(expected_bids, reverse=True)
        if len(sorted_bids) > impression_supply:
            return sorted_bids[impression_supply]
        return sorted_bids[-1] if sorted_bids else 0.0