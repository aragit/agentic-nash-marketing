"""Nash equilibrium solver for multi-agent competitive bidding.

Uses iterative best-response with softmax smoothing and vectorized
Montle Carlo win-probability estimation for performance.
"""

import logging
import numpy as np
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_MC_SAMPLES = 5000


class NashEquilibriumSolver:
    """Computes mixed-strategy Nash equilibrium for N-player ad auction game.

    Each player (brand) chooses a mixed strategy over discrete bid levels.
    The equilibrium is where no player can improve their expected utility
    by unilaterally changing their strategy.
    """

    def __init__(self, bid_levels: List[float] = None):
        self.bid_levels = np.array(
            bid_levels or [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_equilibrium(
        self,
        agent_budgets: Dict[str, float],
        agent_valuations: Dict[str, float],
        impression_supply: int,
        agent_bid_levels: Dict[str, List[float]] = None,
        n_samples: int = DEFAULT_MC_SAMPLES,
    ) -> Dict[str, any]:
        """Compute approximate mixed-strategy Nash equilibrium.

        Uses iterative best-response with softmax smoothing.
        If agent_bid_levels is provided, each agent gets their own bid levels
        (enabling CPA x role differentiated equilibria).
        """
        if agent_bid_levels is None:
            agent_bid_levels = {
                name: self.bid_levels.tolist() for name in agent_budgets
            }

        agent_levels = {name: np.array(levels) for name, levels in agent_bid_levels.items()}
        n_agents = len(agent_budgets)

        if n_agents == 0:
            return {"strategies": {}, "clearing_price": 0.0, "convergence": 0.0}

        strategies = {
            name: np.ones(len(agent_levels[name])) / len(agent_levels[name])
            for name in agent_budgets
        }

        max_iterations = 100
        tolerance = 1e-4
        iteration = 0

        for iteration in range(max_iterations):
            new_strategies = {}

            for agent_name in agent_budgets:
                opponent_names = [n for n in agent_budgets if n != agent_name]
                my_levels = agent_levels[agent_name]

                # Vectorized utility computation for ALL bid levels at once
                utilities = self._vectorized_expected_utilities(
                    my_levels=my_levels,
                    agent_valuations=agent_valuations,
                    agent_name=agent_name,
                    opponent_strategies={n: strategies[n] for n in opponent_names},
                    impression_supply=impression_supply,
                    opponent_levels={n: agent_levels[n] for n in opponent_names},
                    n_samples=n_samples,
                )

                temperature = max(0.1, 1.0 - iteration / max_iterations)
                exp_utils = np.exp(utilities / temperature)
                softmax_strat = exp_utils / np.sum(exp_utils)
                if iteration < max_iterations // 2:
                    epsilon = 0.15 * (1.0 - iteration / (max_iterations // 2))
                    softmax_strat = (1.0 - epsilon) * softmax_strat + epsilon / len(
                        my_levels
                    )
                new_strategies[agent_name] = softmax_strat

            max_diff = max(
                np.max(np.abs(new_strategies[name] - strategies[name]))
                for name in strategies
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
                for name in strategies
            },
            "clearing_price": round(eq_clearing_price, 2),
            "convergence": float(max_diff),
            "iterations": iteration + 1,
        }

    # ------------------------------------------------------------------
    # Vectorized internals
    # ------------------------------------------------------------------

    def _vectorized_expected_utilities(
        self,
        my_levels: np.ndarray,
        agent_valuations: Dict[str, float],
        agent_name: str,
        opponent_strategies: Dict[str, np.ndarray],
        impression_supply: int,
        opponent_levels: Dict[str, np.ndarray],
        n_samples: int = DEFAULT_MC_SAMPLES,
    ) -> np.ndarray:
        """Compute expected utility for ALL candidate bids in one shot.

        Returns:
            Array of shape (n_levels,) with expected utility per bid.
        """
        win_probs = self._vectorized_win_probabilities(
            candidate_bids=my_levels,
            opponent_strategies=opponent_strategies,
            impression_supply=impression_supply,
            opponent_levels=opponent_levels,
            n_samples=n_samples,
        )
        valuation = agent_valuations.get(agent_name, 50.0)
        return win_probs * (valuation - my_levels)

    def _vectorized_win_probabilities(
        self,
        candidate_bids: np.ndarray,
        opponent_strategies: Dict[str, np.ndarray],
        impression_supply: int,
        opponent_levels: Dict[str, np.ndarray],
        n_samples: int = DEFAULT_MC_SAMPLES,
    ) -> np.ndarray:
        """Vectorized win-probability estimation for multiple candidate bids.

        Samples opponent bids in bulk and broadcasts comparisons against
        all candidate bids simultaneously.

        Args:
            candidate_bids: shape (M,) — the bid levels to evaluate.
            opponent_strategies: mapping name → probability vector.
            impression_supply: number of winning slots.
            opponent_levels: mapping name → bid-level array.
            n_samples: Monte Carlo sample count.

        Returns:
            Array of shape (M,) — estimated win probability per candidate bid.
        """
        if not opponent_strategies:
            return np.ones(len(candidate_bids))

        n_opponents = len(opponent_strategies)
        n_levels = len(candidate_bids)

        # 1. Sample opponent bids in bulk: shape (n_opponents, n_samples)
        opp_names = list(opponent_strategies.keys())
        samples = np.zeros((n_opponents, n_samples), dtype=np.float64)
        for j, name in enumerate(opp_names):
            levels = opponent_levels[name]
            probs = opponent_strategies[name]
            # Guard against degenerate distributions
            probs = np.maximum(probs, 0.0)
            prob_sum = probs.sum()
            if prob_sum > 0:
                probs = probs / prob_sum
            else:
                probs = np.ones(len(levels)) / len(levels)
            samples[j] = np.random.choice(levels, size=n_samples, p=probs)

        # 2. Count opponents bidding higher than each candidate bid.
        #    Vectorized across M candidate bids, loop over opponents (cheap).
        M = len(candidate_bids)
        higher_counts = np.zeros((M, n_samples), dtype=np.int32)
        for j in range(n_opponents):
            # samples[j]: (n_samples,) — opponent j's sampled bids
            # candidate_bids[:, None]: (M, 1)
            # Broadcasting gives (M, n_samples): True where opponent j beats candidate
            higher_counts += (samples[j, :] > candidate_bids[:, None]).astype(np.int32)

        # A bid wins iff fewer than `impression_supply` opponents bid higher.
        wins = higher_counts < impression_supply  # (M, n_samples)

        # 4. Win probabilities: shape (M,)
        return np.mean(wins, axis=1)

    # ------------------------------------------------------------------
    # Clearing price
    # ------------------------------------------------------------------

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

        expected_bids = np.array([
            np.dot(strategies[name], agent_levels[name]) for name in strategies
        ])

        sorted_bids = np.sort(expected_bids)[::-1]
        if len(sorted_bids) > impression_supply:
            return float(sorted_bids[impression_supply])
        return float(sorted_bids[-1]) if len(sorted_bids) > 0 else 0.0
