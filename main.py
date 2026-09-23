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
try:
    from .cylindrical_connect_four import (
        WINDOWS,
        CylindricalConnectFour,
        opponent,
        iter_window_cells,
        RED,
        YELLOW,
    )
               
except ImportError:
    from cylindrical_connect_four import (
        WINDOWS,
        CylindricalConnectFour,
        opponent,
        iter_window_cells,
        RED,
        YELLOW,
    )


# Replace this with the name your group wants displayed in the tournament.
GROUP_NAME = "NULL"

# Pre-defined non-linear weights for open window threats
SCORE_3 = 100.0
SCORE_2 = 10.0
SCORE_1 = 1.0

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

    #test first
    MAX_DEPTH = 6

    action, value = alpha_beta(problem, state, float('-inf'), float('inf'), MAX_DEPTH, problem.to_move(state))
    return action


#heuristic that returns a value between -1 and 1 to estimate utility of a non terminal state
def heuristic(game: AdversarialSearchProblem, state: StateT, player: PlayerT) -> float:
    # from cylindrical_connect_four import RED, YELLOW
    if player == RED:
        opponent = YELLOW
    else:
        opponent = RED

    total_score = 0.0

    # Iterate over all 4-cell windows on the board
    for window_cells in iter_window_cells(state):
        player_pieces = 0
        opponent_pieces = 0

        for cell in window_cells:
            if (cell == opponent):
                opponent_pieces += 1 # Keep positive for simplicity
            elif (cell == player):
                player_pieces += 1 

        # Skip windows containing pieces from BOTH players (neither can win this window)
        if player_pieces > 0 and opponent_pieces > 0:
            continue

        if player_pieces > 0:
            if player_pieces == 3:
                total_score += SCORE_3
            elif player_pieces == 2:
                total_score += SCORE_2
            elif player_pieces == 1:
                total_score += SCORE_1
        elif opponent_pieces > 0:
            if opponent_pieces == 3:
                total_score -= SCORE_3
            elif opponent_pieces == 2:
                total_score -= SCORE_2
            elif opponent_pieces == 1:
                total_score -= SCORE_1

    return total_score
    

    # # for every possible combination of four-tile-windows
    # # runs for all window cells. Works by setting window_cells to next and checking if it is None
    # while (window_cells := next(iterable, None)) is not None:

    #     if window_cells is None:
    #         break

    #     # count player pieces as positive and opponent as negative
    #     player_pieces = 0
    #     opponent_pieces = 0
    #     empty_cells = 0
        
    #     window_value = 0
    #     for cell in window_cells:
    #         if (cell == opponent):
    #             opponent_pieces -= 1
    #         elif (cell == player):
    #             player_pieces += 1
    #         # else:
    #         #     empty_cells += 1

    #     #if there are both color pieces in this window, there is no way this window can be won by a player
    #     if player_pieces > 0 and opponent_pieces < 0:
    #         continue
    #     # now this window only consists of one color and possibly empty spaces

    #     utility = 0
    #     if player_pieces > 0:
    #         if player_pieces == 3:
    #             utility += SCORE_3
    #         elif player_pieces == 2:
    #             utility += SCORE_2
    #         elif player_pieces == 1:
    #             utility += SCORE_1
    #     elif opponent_pieces < 0:
    #         if opponent_pieces == -3:
    #             utility -= SCORE_3
    #         elif opponent_pieces == -2:
    #             utility -= SCORE_2
    #         elif opponent_pieces == -1:
    #             utility -= SCORE_1

        
    #     # # pieces will be either all opponent pieces or player pieces since one is 0
    #     # pieces = player_pieces + opponent_pieces

    #     # # utility is the utility of this window. Will be positive if it is players' pieces and negative if it is opponents'
    #     # utility = pieces * 0.25

    #     # #return 1 or -1 if there are 4 pieces in the window
    #     # if abs(pieces) == 4:
    #     #     return utility

    #     # assign best player or opponent by comparing with best so far
    #     if utility >= 0:
    #         best_player = max(best_player, utility)
    #     else:
    #         best_opponent = min(best_opponent, utility)

    #     #best won't get any better than this
    #     if best_player == -111.0 and best_opponent == 111.0:
    #         break

    # # return best of both (Ex. 3 pieces is the player's best and 3 is the opponent's best, heuristic of this state would be 0)
    # return best_player + best_opponent

def get_ordered_actions(game: AdversarialSearchProblem, state: StateT) -> list[ActionT]:
    #Orders legal actions starting from center columns outwards to maximize Alpha-Beta cutoffs
    actions = list(game.actions(state))
    board_width = getattr(game, "columns", 7)
    center = board_width / 2.0
    actions.sort(key=lambda col: abs(col - center))
    return actions

def alpha_beta(game: AdversarialSearchProblem[StateT, ActionT, PlayerT], state: StateT, alpha, beta, depth, player : PlayerT) -> tuple[ActionT | None, float]:
    # if the game is a terminal state, return just the utility
    if game.is_terminal(state):
        return None, game.utility(state, player)

    # if this iteration depth is at 0, then don't continue and return a heuristic estimate of the state
    if depth == 0:
        return None, heuristic(game, state, player)

    # set up variables to store and compare to find the best action and its associated utility
    
    best_action = None
    max_util = float('-inf')
    min_util = float('inf')
    max_player_bool = game.to_move(state) == player


    # check children of current node to find best
    for action in get_ordered_actions(game, state):
        #if Max player
        if max_player_bool:
        # get the child utility for this action
            _ , child_util = alpha_beta(game, game.result(state, action), alpha, beta, depth - 1, player)

            # if child util is greater than the current maximum, set max to child util and action
            if child_util > max_util:
                best_action = action
                max_util = child_util

            # if max util is greater than beta, no point in checking rest of children so return now
            if max_util >= beta:
                return best_action, max_util

            # update alpha for next children
            alpha = max(alpha, max_util)   
        #if MIN player
        else:
        # get the child utility for this action
            _ , child_util = alpha_beta(game, game.result(state, action), alpha, beta, depth - 1, player)
    
            # if child util is less than the current minimum, set min to child util and action
            if child_util < min_util:
                best_action = action
                min_util = child_util
    
            # if min util is less than alpha, no point in checking rest of children so return now
            if min_util <= alpha:
                return best_action, min_util
    
            # update beta for next children
            beta = min(beta, min_util)
            
    return best_action, max_util if max_player_bool else min_util
    

        

    