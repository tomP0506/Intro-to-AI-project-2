"""Public tests for CS 4341 Assignment 2.

The tests intentionally do not require one particular move or Chapter 5
algorithm.  They check the common contract: legal and timely actions,
terminal handling, and a complete game.
"""

from __future__ import annotations

import random
import signal
import time
from contextlib import contextmanager
from typing import Iterator

import pytest

try:
    from . import main
    from .cylindrical_connect_four import (
        YELLOW,
        Action,
        CylindricalConnectFour,
        GameState,
        Player,
        format_state,
        state_from_actions,
    )
except ImportError:
    import main
    from cylindrical_connect_four import (
        YELLOW,
        Action,
        CylindricalConnectFour,
        GameState,
        Player,
        format_state,
        state_from_actions,
    )


# Actions are zero-based column numbers. These five fixed sequences produce
# distinct, legal, nonterminal positions.
PUBLIC_ACTION_SEQUENCES = (
    (3, 2, 4, 3, 2, 4, 0),
    (0, 6, 1, 5, 6, 0, 5, 1),
    (6, 0, 5, 1, 0, 6, 1, 5, 3, 2, 2),
    (2, 3, 2, 4, 3, 4, 5, 1, 1, 5),
    (4, 5, 3, 4, 5, 3, 0, 6, 0, 6, 2, 1),
)
MOVE_TIME_LIMIT_SECONDS = 10.0
MAX_GAME_PLIES = 42
RANDOM_OPPONENT_SEED = 434_106 ^ 0x5A17
_PLACEHOLDER_GROUP_NAMES = {
    "",
    "replace-with-your-group-name",
    "todo",
    "group name",
}


class _MoveTimeout(Exception):
    pass


@contextmanager
def _time_limit(seconds: float) -> Iterator[None]:
    """Interrupt a student call that reaches the per-move limit."""
    if not hasattr(signal, "setitimer"):
        yield
        return

    def timeout_handler(signum: int, frame: object) -> None:
        del signum, frame
        raise _MoveTimeout

    previous_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)


def _call_agent(
    problem: CylindricalConnectFour,
    state: GameState,
) -> tuple[Action | None, float]:
    started = time.perf_counter()
    try:
        with _time_limit(MOVE_TIME_LIMIT_SECONDS):
            action = main.adversarial_search(problem, state)
    except _MoveTimeout:
        pytest.fail(
            "adversarial_search reached the 10-second per-move limit"
        )
    elapsed = time.perf_counter() - started
    assert elapsed < MOVE_TIME_LIMIT_SECONDS, (
        "adversarial_search must take less than 10 seconds per move; "
        f"this call took {elapsed:.3f} seconds"
    )
    return action, elapsed


def _public_state(case_index: int) -> GameState:
    return state_from_actions(PUBLIC_ACTION_SEQUENCES[case_index])


def test_group_name_is_set() -> None:
    group_name = getattr(main, "GROUP_NAME", None)
    assert isinstance(group_name, str), "GROUP_NAME must be a string"
    assert group_name.strip().lower() not in _PLACEHOLDER_GROUP_NAMES, (
        "replace GROUP_NAME with the name your group will use in the tournament"
    )


def test_provided_positions_are_distinct_nonterminal_states() -> None:
    states = {_public_state(index) for index in range(5)}
    assert len(states) == 5
    for state in states:
        problem = CylindricalConnectFour(state)
        assert not problem.is_terminal(state)
        assert problem.actions(state)


def test_board_display_repeats_columns_beyond_the_seam() -> None:
    state = state_from_actions((0, 6, 1))
    display = format_state(state)
    assert "1   2   3   4   5   6   7 : 1   2   3" in display
    assert display.count(" : ") == 9


@pytest.mark.parametrize(
    "case_index",
    range(5),
    ids=lambda index: f"position_{index + 1}",
)
def test_search_returns_timely_legal_action(case_index: int) -> None:
    state = _public_state(case_index)
    problem = CylindricalConnectFour(state)
    legal_actions = problem.actions(state)
    action, _ = _call_agent(problem, state)
    assert type(action) is int and action in legal_actions, (
        "adversarial_search must return an integer action from "
        f"problem.actions(state); got {action!r}"
    )


def test_search_returns_none_for_terminal_state() -> None:
    state = state_from_actions((0, 1, 0, 1, 0, 1, 0))
    problem = CylindricalConnectFour(state)
    assert problem.is_terminal(state)

    action, _ = _call_agent(problem, state)
    assert action is None, (
        "adversarial_search must return None for a terminal state; "
        f"got {action!r}"
    )


def test_agent_completes_game_against_random_opponent() -> None:
    state = _public_state(4)
    problem = CylindricalConnectFour(state)
    student_side: Player = YELLOW
    rng = random.Random(RANDOM_OPPONENT_SEED)

    while not problem.is_terminal(state) and state.ply < MAX_GAME_PLIES:
        if problem.to_move(state) == student_side:
            action, _ = _call_agent(problem, state)
            assert type(action) is int and action in problem.actions(state), (
                "adversarial_search returned a non-integer or illegal action "
                f"during a game: {action!r}"
            )
        else:
            action = rng.choice(problem.actions(state))
        state = problem.result(state, action)

    assert problem.is_terminal(state), (
        f"the game did not finish within {MAX_GAME_PLIES} half-moves"
    )
