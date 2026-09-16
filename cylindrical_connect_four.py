"""Cylindrical Connect Four as an adversarial-search problem.

The board has seven columns and six rows.  Pieces fall under gravity exactly
as they do in ordinary Connect Four, but the left and right edges touch: a
horizontal or diagonal line may continue from column 7 into column 1.

Columns in the Python interface are zero-based.  The terminal display uses
the human-friendly labels 1 through 7.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, Literal, TypeAlias


# ---------------------------------------------------------------------------
# Public types and board constants
# ---------------------------------------------------------------------------

Player: TypeAlias = Literal["red", "yellow"]
Cell: TypeAlias = Player | None
Action: TypeAlias = int
Position: TypeAlias = tuple[int, int]

RED: Player = "red"
YELLOW: Player = "yellow"
COLUMNS = 7
ROWS = 6
CONNECT = 4


def opponent(player: Player) -> Player:
    """Return the other player."""
    if player == RED:
        return YELLOW
    if player == YELLOW:
        return RED
    raise ValueError(f"Unknown player: {player!r}")


def _make_windows() -> tuple[tuple[Position, ...], ...]:
    """Build every four-cell winning window on the cylindrical board."""
    windows: list[tuple[Position, ...]] = []

    # Vertical lines do not wrap.
    for column in range(COLUMNS):
        for start_row in range(ROWS - CONNECT + 1):
            windows.append(
                tuple(
                    (column, start_row + offset)
                    for offset in range(CONNECT)
                )
            )

    # Horizontal and diagonal lines wrap around the cylinder.
    for row in range(ROWS):
        for start_column in range(COLUMNS):
            windows.append(
                tuple(
                    ((start_column + offset) % COLUMNS, row)
                    for offset in range(CONNECT)
                )
            )
    for start_row in range(ROWS - CONNECT + 1):
        for start_column in range(COLUMNS):
            windows.append(
                tuple(
                    (
                        (start_column + offset) % COLUMNS,
                        start_row + offset,
                    )
                    for offset in range(CONNECT)
                )
            )
    for start_row in range(CONNECT - 1, ROWS):
        for start_column in range(COLUMNS):
            windows.append(
                tuple(
                    (
                        (start_column + offset) % COLUMNS,
                        start_row - offset,
                    )
                    for offset in range(CONNECT)
                )
            )
    return tuple(windows)


# Each position is ``(column, row)`` with zero-based coordinates.  Exposing
# the windows keeps heuristics from having to reimplement the wraparound rule.
WINDOWS: tuple[tuple[Position, ...], ...] = _make_windows()


@dataclass(frozen=True, slots=True)
class CylindricalConnectFourState:
    """An immutable, hashable snapshot of a game.

    ``columns[column]`` is a tuple containing that column's pieces from the
    bottom row upward.  This representation encodes gravity without empty
    holes.  ``player`` is the player whose turn comes next.
    """

    columns: tuple[tuple[Player, ...], ...] = ((),) * COLUMNS
    player: Player = RED

    def __post_init__(self) -> None:
        if len(self.columns) != COLUMNS:
            raise ValueError(f"A board must have exactly {COLUMNS} columns.")
        for column in self.columns:
            if not isinstance(column, tuple):
                raise TypeError("Each column must be a tuple.")
            if len(column) > ROWS:
                raise ValueError(f"A column may contain at most {ROWS} pieces.")
            if any(cell not in (RED, YELLOW) for cell in column):
                raise ValueError("Board cells must contain 'red' or 'yellow'.")
        if self.player not in (RED, YELLOW):
            raise ValueError(f"Unknown player: {self.player!r}")

    @property
    def ply(self) -> int:
        """Return the number of pieces that have been played."""
        return sum(len(column) for column in self.columns)

    @property
    def winner(self) -> Player | None:
        """Return the player with a four-in-a-row, if one exists."""
        winners = {
            first
            for window in WINDOWS
            if (first := self.cell(*window[0])) is not None
            and all(self.cell(*position) == first for position in window[1:])
        }
        if len(winners) > 1:
            raise ValueError("The board contains winning lines for both players.")
        return next(iter(winners), None)

    @property
    def is_full(self) -> bool:
        """Return whether all 42 cells are occupied."""
        return self.ply == COLUMNS * ROWS

    def cell(self, column: int, row: int) -> Cell:
        """Return one cell; row 0 is the bottom of the board."""
        if not 0 <= column < COLUMNS or not 0 <= row < ROWS:
            raise IndexError("Cell coordinates are outside the board.")
        stack = self.columns[column]
        return stack[row] if row < len(stack) else None

    def __str__(self) -> str:
        return format_state(self)


GameState: TypeAlias = CylindricalConnectFourState


# ---------------------------------------------------------------------------
# Adversarial-search problem
# ---------------------------------------------------------------------------

class CylindricalConnectFour:
    """A deterministic, two-player, zero-sum game."""

    def __init__(self, initial: GameState | None = None) -> None:
        self.initial = initial if initial is not None else GameState()
        if not isinstance(self.initial, CylindricalConnectFourState):
            raise TypeError("initial must be a CylindricalConnectFourState.")

    def to_move(self, state: GameState) -> Player:
        """Return the player whose turn it is."""
        self._require_state(state)
        return state.player

    def actions(self, state: GameState) -> tuple[Action, ...]:
        """Return legal zero-based column numbers in ascending order."""
        self._require_state(state)
        if self.is_terminal(state):
            return ()
        return tuple(
            column
            for column, stack in enumerate(state.columns)
            if len(stack) < ROWS
        )

    def result(self, state: GameState, action: Action) -> GameState:
        """Return the state after dropping a piece in ``action``."""
        self._require_state(state)
        if self.is_terminal(state):
            raise ValueError("A terminal game state cannot be changed.")
        if type(action) is not int:
            raise TypeError("An action must be an integer column from 0 to 6.")
        if action not in self.actions(state):
            raise ValueError(f"Column {action!r} is not legal in this state.")

        columns = list(state.columns)
        columns[action] = columns[action] + (state.player,)
        return GameState(tuple(columns), opponent(state.player))

    def is_terminal(self, state: GameState) -> bool:
        """Return whether the state is a win or a full-board draw."""
        self._require_state(state)
        return state.winner is not None or state.is_full

    def utility(self, state: GameState, player: Player) -> float:
        """Return +1 for a win, -1 for a loss, and 0 for a draw."""
        self._require_state(state)
        if player not in (RED, YELLOW):
            raise ValueError(f"Unknown player: {player!r}")
        if not self.is_terminal(state):
            raise ValueError("Utility is defined only for terminal states.")
        if state.winner is None:
            return 0.0
        return 1.0 if state.winner == player else -1.0

    @staticmethod
    def _require_state(state: object) -> None:
        if not isinstance(state, CylindricalConnectFourState):
            raise TypeError("Expected a CylindricalConnectFourState.")


# ---------------------------------------------------------------------------
# Construction, serialization, and display helpers
# ---------------------------------------------------------------------------

def state_from_actions(
    actions: Iterable[Action],
    initial: GameState | None = None,
) -> GameState:
    """Apply a legal action sequence and return the resulting state."""
    problem = CylindricalConnectFour(initial)
    state = problem.initial
    for index, action in enumerate(actions):
        if problem.is_terminal(state):
            raise ValueError(
                f"The game ended before action {index + 1} in the sequence."
            )
        state = problem.result(state, action)
    return state


def iter_window_cells(
    state: GameState,
) -> Iterator[tuple[Cell, Cell, Cell, Cell]]:
    """Yield the contents of every vertical or wrapped winning window."""
    if not isinstance(state, CylindricalConnectFourState):
        raise TypeError("Expected a CylindricalConnectFourState.")
    for window in WINDOWS:
        cells = tuple(state.cell(*position) for position in window)
        yield cells  # type: ignore[misc]


def state_to_dict(state: GameState) -> dict[str, object]:
    """Serialize a state for the autograder and tournament protocols."""
    if not isinstance(state, CylindricalConnectFourState):
        raise TypeError("Expected a CylindricalConnectFourState.")
    return {
        "columns": [list(column) for column in state.columns],
        "player": state.player,
    }


def state_from_dict(data: object) -> GameState:
    """Deserialize and validate a state."""
    if not isinstance(data, dict):
        raise TypeError("A serialized state must be an object.")
    raw_columns = data.get("columns")
    player = data.get("player")
    if not isinstance(raw_columns, list):
        raise TypeError("A serialized state must contain a columns list.")
    columns: list[tuple[Player, ...]] = []
    for raw_column in raw_columns:
        if not isinstance(raw_column, list):
            raise TypeError("Each serialized column must be a list.")
        columns.append(tuple(raw_column))  # type: ignore[arg-type]
    return GameState(tuple(columns), player)  # type: ignore[arg-type]


def action_to_dict(action: Action) -> dict[str, int]:
    """Serialize one zero-based column action."""
    if type(action) is not int or not 0 <= action < COLUMNS:
        raise ValueError("An action must be an integer column from 0 to 6.")
    return {"column": action}


def action_from_dict(data: object) -> Action:
    """Deserialize one zero-based column action."""
    if not isinstance(data, dict):
        raise TypeError("A serialized action must be an object.")
    column = data.get("column")
    if type(column) is not int or not 0 <= column < COLUMNS:
        raise ValueError("A serialized action needs an integer column 0 to 6.")
    return column


def format_state(state: GameState) -> str:
    """Render the board with columns 1--3 repeated beyond the dotted seam."""
    if not isinstance(state, CylindricalConnectFourState):
        raise TypeError("Expected a CylindricalConnectFourState.")

    symbols: dict[Cell, str] = {None: ".", RED: "X", YELLOW: "O"}
    repeated = range(CONNECT - 1)
    left_header = "    " + "   ".join(str(column) for column in range(1, 8))
    right_header = "   ".join(str(column) for column in range(1, CONNECT))
    left_border = "  +" + "---+" * COLUMNS
    right_border = "+" + "---+" * (CONNECT - 1)
    divider = " : "
    lines = [
        f"{left_header}{divider}{right_header}",
        f"{left_border}{divider}{right_border}",
    ]
    for row in range(ROWS - 1, -1, -1):
        left_cells = " | ".join(
            symbols[state.cell(column, row)]
            for column in range(COLUMNS)
        )
        right_cells = " | ".join(
            symbols[state.cell(column, row)]
            for column in repeated
        )
        lines.append(
            f"{row + 1} | {left_cells} |{divider}| {right_cells} |"
        )
    lines.append(f"{left_border}{divider}{right_border}")

    if state.winner is not None:
        status = f"Winner: {state.winner.title()} ({symbols[state.winner]})"
    elif state.is_full:
        status = "Draw: the board is full."
    else:
        status = f"To move: {state.player.title()} ({symbols[state.player]})"
    lines.append(status)
    return "\n".join(lines)


__all__ = [
    "Action",
    "COLUMNS",
    "CONNECT",
    "Cell",
    "CylindricalConnectFour",
    "CylindricalConnectFourState",
    "GameState",
    "Player",
    "Position",
    "RED",
    "ROWS",
    "WINDOWS",
    "YELLOW",
    "action_from_dict",
    "action_to_dict",
    "format_state",
    "iter_window_cells",
    "opponent",
    "state_from_actions",
    "state_from_dict",
    "state_to_dict",
]
