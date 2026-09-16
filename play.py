#!/usr/bin/env python3
"""Play a student's Cylindrical Connect Four agent in the terminal."""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable, Sequence

try:
    from . import main as student_main
    from .cylindrical_connect_four import (
        COLUMNS,
        RED,
        YELLOW,
        CylindricalConnectFour,
        GameState,
        Player,
        format_state,
        opponent,
    )
except ImportError:
    import main as student_main
    from cylindrical_connect_four import (
        COLUMNS,
        RED,
        YELLOW,
        CylindricalConnectFour,
        GameState,
        Player,
        format_state,
        opponent,
    )


InputFunction = Callable[[str], str]
OutputFunction = Callable[[str], None]


def _human_action(
    problem: CylindricalConnectFour,
    state: GameState,
    input_function: InputFunction,
    output: OutputFunction,
) -> int | None:
    """Prompt until the human selects a legal column or quits."""
    while True:
        answer = input_function("Choose a column (1-7), or q to quit: ").strip()
        if answer.lower() in {"q", "quit", "exit"}:
            return None
        try:
            action = int(answer) - 1
        except ValueError:
            output("Enter a column number from 1 through 7.")
            continue
        if not 0 <= action < COLUMNS:
            output("Enter a column number from 1 through 7.")
        elif action not in problem.actions(state):
            output(f"Column {action + 1} is full; choose another column.")
        else:
            return action


def run_game(
    human_player: Player = RED,
    *,
    input_function: InputFunction = input,
    output: OutputFunction = print,
) -> int:
    """Run one human-versus-student-agent game."""
    if human_player not in (RED, YELLOW):
        raise ValueError("human_player must be 'red' or 'yellow'.")

    problem = CylindricalConnectFour()
    state = problem.initial
    bot_player = opponent(human_player)
    marks = {RED: "X", YELLOW: "O"}

    output("Cylindrical Connect Four")
    output("The dotted line is the seam; columns 1-3 repeat on its right.")
    output(
        f"You are {human_player.title()} ({marks[human_player]}); "
        f"your bot is {bot_player.title()} ({marks[bot_player]})."
    )

    while not problem.is_terminal(state):
        output("\n" + format_state(state))
        if problem.to_move(state) == human_player:
            action = _human_action(
                problem,
                state,
                input_function,
                output,
            )
            if action is None:
                output("Game ended by the player.")
                return 0
        else:
            output("Your bot is thinking...")
            started = time.perf_counter()
            try:
                action = student_main.adversarial_search(
                    problem,
                    state,
                )
            except Exception as error:
                output(
                    "Your bot raised "
                    f"{type(error).__name__}: {error}"
                )
                output(f"{human_player.title()} wins by resignation.")
                return 1
            elapsed = time.perf_counter() - started
            if type(action) is not int or action not in problem.actions(state):
                output(f"Your bot returned an illegal action: {action!r}")
                output(f"{human_player.title()} wins by resignation.")
                return 1
            output(
                f"Your bot chose column {action + 1} "
                f"in {elapsed:.3f} seconds."
            )
        state = problem.result(state, action)

    output("\n" + format_state(state))
    if state.winner is None:
        output("Game over: draw.")
    elif state.winner == human_player:
        output("Game over: you win!")
    else:
        output("Game over: your bot wins!")
    return 0


def _arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Play your Assignment 2 agent in the terminal.",
    )
    parser.add_argument(
        "--human",
        choices=(RED, YELLOW),
        default=RED,
        help="Human side; Red moves first (default: red).",
    )
    return parser.parse_args(argv)


def cli(argv: Sequence[str] | None = None) -> int:
    arguments = _arguments(argv)
    return run_game(arguments.human)


if __name__ == "__main__":
    raise SystemExit(cli())
