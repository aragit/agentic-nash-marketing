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
        self.bid_levels = np.array(bid_levels or [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])

    def compute_equilibrium(
        self,
        agent_budgets: Dict[str, float],
        agent_valuations: Dict[str, float],
        impression_supply: int,
        agent_bid_levels: Dict[str, List[float]] = None,
    ) -> Dict[str, any]:
        """
        Compute approximate mixed-strategy Nash equilibrium.

        Uses iterative best-response with softmax smoothing.
        If agent_bid_levels is provided, each agent gets their own bid levels
        (enabling CPA × role differentiated equilibria).
        """
        if agent_bid_levels is None:
            agent_bid_levels = {name: self.bid_levels.tolist() for name in agent_budgets}

        # Convert to numpy arrays
        agent_levels = {name: np.array(levels) for name, levels in agent_bid_levels.items()}
        n_agents = len(agent_budgets)

        if n_agents == 0:
            return {"strategies": {}, "clearing_price": 0.0, "convergence": 0.0}

        # Initialize uniform mixed strategies over each agent's own bid levels
        strategies = {
            name: np.ones(len(agent_levels[name])) / len(agent_levels[name])
            for name in agent_budgets.keys()
        }

        max_iterations = 100
        tolerance = 1e-4

        for iteration in range(max_iterations):
            new_strategies = {}

            for agent_name in agent_budgets.keys():
                opponent_names = [n for n in agent_budgets.keys() if n != agent_name]
                my_levels = agent_levels[agent_name]
                n_levels = len(my_levels)

                utilities = np.zeros(n_levels)

                for i, bid in enumerate(my_levels):
                    expected_utility = self._expected_utility(
                        agent_name=agent_name,
                        bid=bid,
                        agent_budgets=agent_budgets,
                        agent_valuations=agent_valuations,
                        opponent_strategies={
                            n: strategies[n] for n in opponent_names
                        },
                        impression_supply=impression_supply,
                        opponent_levels={n: agent_levels[n] for n in opponent_names},
                    )
                    utilities[i] = expected_utility

                temperature = max(0.1, 1.0 - iteration / max_iterations)
                exp_utils = np.exp(utilities / temperature)
                softmax_strat = exp_utils / np.sum(exp_utils)
                if iteration < max_iterations // 2:
                    epsilon = 0.15 * (1.0 - iteration / (max_iterations // 2))
                    softmax_strat = (1.0 - epsilon) * softmax_strat + epsilon / n_levels
                new_strategies[agent_name] = softmax_strat

            max_diff = max(
                np.max(np.abs(new_strategies[name] - strategies[name]))
                for name in strategies.keys()
            )

            strategies = new_strategies

            if max_diff < tolerance:
                logger.info(f"Nash equilibrium converged in {iteration + 1} iterations")
                break

        eq_clearing_price = self._equilibrium_clearing_price(
            strategies, agent_budgets, impression_supply, agent_levels
        )

        return {
            "strategies": {
                name: {
                    "distribution": strategies[name].tolist(),
                    "expected_bid": float(np.dot(strategies[name], agent_levels[name])),
                    "bid_levels": agent_levels[name].tolist(),
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
        opponent_levels: Dict[str, np.ndarray] = None,
    ) -> float:
        """Compute expected utility for a single bid level."""
        valuation = agent_valuations.get(agent_name, 50.0)
        win_prob = self._win_probability(bid, opponent_strategies, impression_supply, opponent_levels)
        return win_prob * (valuation - bid)

    def _win_probability(
        self,
        bid: float,
        opponent_strategies: Dict[str, np.ndarray],
        impression_supply: int,
        opponent_levels: Dict[str, np.ndarray] = None,
    ) -> float:
        """Stochastic win probability via Monte Carlo."""
        if not opponent_strategies:
            return 1.0

        if opponent_levels is None:
            opponent_levels = {name: self.bid_levels for name in opponent_strategies}

        n_samples = 5000
        n_opponents = len(opponent_strategies)
        samples = np.zeros((n_samples, n_opponents))
        for j, (name, strategy) in enumerate(opponent_strategies.items()):
            levels = opponent_levels[name]
            samples[:, j] = np.random.choice(levels, size=n_samples, p=strategy)

        higher_bids = np.sum(samples > bid, axis=1)
        wins = np.sum(higher_bids < impression_supply)
        return float(wins) / n_samples

    def _equilibrium_clearing_price(
        self,
        strategies: Dict[str, np.ndarray],
        agent_budgets: Dict[str, float],
        impression_supply: int,
        agent_levels: Dict[str, np.ndarray] = None,
    ) -> float:
        """Estimate equilibrium clearing price from mixed strategies."""
        if agent_levels is None:
            agent_levels = {name: self.bid_levels for name in strategies}

        expected_bids = []
        for name, strategy in strategies.items():
            expected_bid = np.dot(strategy, agent_levels[name])
            expected_bids.append(expected_bid)

        sorted_bids = sorted(expected_bids, reverse=True)
        if len(sorted_bids) > impression_supply:
            return sorted_bids[impression_supply]
        return sorted_bids[-1] if sorted_bids else 0.0