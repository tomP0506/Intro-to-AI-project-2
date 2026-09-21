"""Shared adversarial search interfaces for CS 4341 Assignment 2."""
from __future__ import annotations

# Import packages
from typing import Hashable, Iterable, Protocol, TypeVar

# ---------------------------------------------------------------------------
# Type variables
# ---------------------------------------------------------------------------

StateT = TypeVar("StateT", bound=Hashable)
ActionT = TypeVar("ActionT", bound=Hashable)
PlayerT = TypeVar("PlayerT", bound=Hashable)

# ---------------------------------------------------------------------------
# Adversarial search problem interface
# ---------------------------------------------------------------------------

class AdversarialSearchProblem(
    Protocol[StateT, ActionT, PlayerT]
):
    """
    Interface required by minimax and Monte Carlo tree search.

    The problem represents a deterministic, two-player, zero-sum game.

    States, actions, and players must be hashable because the algorithms may
    use them as dictionary keys.
    """

    initial: StateT

    def to_move(self, state: StateT) -> PlayerT:
        """Return the player whose turn it is in state."""
        ...

    def actions(self, state: StateT) -> Iterable[ActionT]:
        """Return the legal actions available in state."""
        ...

    def result(self, state: StateT, action: ActionT) -> StateT:
        """Return the state produced by applying action in state."""
        ...

    def is_terminal(self, state: StateT) -> bool:
        """Return True exactly when state is terminal."""
        ...

    def utility(self, state: StateT, player: PlayerT) -> float:
        """
        Return the utility of terminal state for player.

        A larger value is better for player. Because the game is zero-sum,
        the two players receive opposite utilities.
        """
        ...