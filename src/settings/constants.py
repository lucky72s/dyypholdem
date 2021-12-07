from enum import Enum

# --- the number of players in the game
players_count = 2
# --- the number of betting rounds in the game
streets_count = 4


# --- IDs for each player and chance
# -- @field chance `-1`
# -- @field P1 `0`
# -- @field P2 `1`
class Players(Enum):
    Chance = -1
    P1 = 0
    P2 = 1

    def __repr__(self):
        if self is Players.P1:
            return "P1 (SB)"
        else:
            return "P2 (BB)"


# --- IDs for terminal nodes (either after a fold or call action) and nodes that follow a check action
# -- @field terminal_fold (terminal node following fold) `-2`
# -- @field terminal_call (terminal node following call) `-1`
# -- @field chance_node (node for the chance player) `0`
# -- @field check (node following check) `-1`
# -- @field inner_node (any other node) `1`
class NodeTypes(Enum):
    terminal_fold = -2
    terminal_call = -1
    inner_check = -1
    chance_node = 0
    inner_raise = 1
    undefined = -9


# --- IDs for fold and check/call actions
# -- @field fold `-2`
# -- @field ccall (check/call) `-1`
class Actions(Enum):
    fold = -2
    ccall = -1
    rraise = -3


# --- String representations for actions in the ACPC protocol
# -- @field fold "`fold`"
# -- @field ccall (check/call) "`ccall`"
# -- @field raise "`raise`"
class ACPCActions(Enum):
    fold = "fold"
    ccall = "ccall"
    rraise = "raise"


# --- An arbitrarily large number used for clamping regrets.
# --@return the number
def max_number():
    return 999999
