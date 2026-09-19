"""Student implementations for CS 4341 Assignment 2."""
from __future__ import annotations

# Import utilities
try:
    from .adversarial_search import (
        ActionT,
        AdversarialSearchProblem,
        PlayerT,
        StateT,
    )
except ImportError:
    from adversarial_search import (
        ActionT,
        AdversarialSearchProblem,
        PlayerT,
        StateT,
    )


# Replace this with the name your group wants displayed in the tournament.
GROUP_NAME = "replace-with-your-group-name"


def adversarial_search(
    problem: AdversarialSearchProblem[StateT, ActionT, PlayerT],
    state: StateT,
) -> ActionT | None:
    """
    Choose an action for the current game state.

    How your agent makes this choice is up to you.  You may add any helpers
    you find useful to this file, such as a heuristic, quiescence search,
    rollout policy, move ordering, or a cache.  None of those helpers is a
    required part of the submission interface.

    The ``problem`` object provides the game rules through methods including
    ``actions``, ``result``, ``to_move``, ``is_terminal``, and ``utility``.
    Your agent is responsible for choosing and managing its own search limits.

    Args:
        problem:
            The adversarial search problem.

        state:
            The current game state.

    Returns:
        The selected action: a zero-based column number from
        ``problem.actions(state)``.
        None if state is terminal or has no legal actions.

    Performance:
        Every call must return in less than 10 seconds.  Choose an internal
        search budget with enough margin to satisfy that hard limit.
    """
    MAX_DEPTH = 10
    action, value = max_value(problem, state, float('-inf'), float('inf'), MAX_DEPTH)
    return action
    # raise NotImplementedError

#plays from the MAX perspective and returns a tuple of an action and its utility
def max_value(game: AdversarialSearchProblem, state: StateT, alpha, beta, depth) -> tuple[ActionT | None, float]:
    # if the game is a terminal state, return just the utility
    if game.is_terminal(state):
        return None, game.utility(state, game.to_move(state))

    # if this iteration depth is at 0, then don't continue and return a heuristic estimate of the state
    if depth == 0:
        return None, heuristic(game, state)

    # set up variables to store and compare to find the best action and its associated utility
    best_action = None
    max_util = float('-inf')

    # check children of current node to find best
    for action in game.actions(state):
        # get the child utility for this action
        _ , child_util = min_value(game, game.result(state), alpha, beta, depth - 1)

        # if child util is greater than the current maximum, set max to child util and action
        if child_util > max_util:
            best_action = action
            max_util = child_util

        # if max util is greater than beta, no point in checking rest of children so return now
        if max_util >= beta:
            return best_action, max_util

        # update alpha for next children
        alpha = max(alpha, max_util)

    # return best from the children
    return best_action, max_util

# plays from the MIN perspective and returns a tuple of an action and its utility
def min_value(game: AdversarialSearchProblem, state: StateT, alpha, beta, depth) -> tuple[ActionT | None, float]:
    # if the game is a terminal state, return just the utility
    if game.is_terminal(state):
        return None, game.utility(state, game.to_move(state))

    # if this iteration depth is at 0, then don't continue and return a heuristic estimate of the state
    if depth == 0:
        return None, heuristic(game, state)

    # set up variables to store and compare to find the best action and its associated utility
    best_action = None
    min_util = float('inf')

    # check children of current node to find best
    for action in game.actions(state):
        # get the child utility for this action
        _ , child_util = max_value(game, game.result(state), alpha, beta, depth - 1)

        # if child util is greater than the current maximum, set max to child util and action
        if child_util > min_util:
            best_action = action
            min_util = child_util

        # if max util is greater than beta, no point in checking rest of children so return now
        if min_util <= alpha:
            return best_action, min_util

        # update alpha for next children
        alpha = min(beta, min_util)

    # return best from the children
    return best_action, min_util
    # raise NotImplementedError


#heuristic that returns a value between -1 and 1 to estimate utility of a non terminal state
def heuristic(game: AdversarialSearchProblem, state: StateT) -> float:
    raise NotImplementedError
