import re
from dataclasses import dataclass

import settings.arguments as arguments
import settings.constants as constants
import settings.game_settings as game_settings

from tree.tree_node import TreeNode
import game.card_tools as card_tools
import game.card_to_string_conversion as card_to_string


@dataclass
class ParsedState(object):
    position: int = -1
    hand_id: int = 0
    actions: list = None
    actions_raw: list = None
    board: str = ""
    hand_p1: str = ""
    hand_p2: str = ""


@dataclass
class ProcessedState(object):
    hand_number: int = 0
    position: int = 0
    player: constants.Players = constants.Players.Chance
    current_street: int = 0
    actions: list = None
    actions_raw: list = None
    all_actions: list = None
    board: str = ""
    my_hand_string: str = ""
    opponent_hand_string: str = ""
    hand_id: int = 0
    acting_player: constants.Players = constants.Players.Chance
    bet1: int = 0
    bet2: int = 0

    def __repr__(self):
        line_1 = f"{arguments.street_names[self.current_street]} - Position: {self.position} / {repr(self.player)} - Pocket cards: {self.my_hand_string} - Board: {self.board} - "
        if self.actions[self.current_street-1]:
            last_action = self.actions[self.current_street-1][len(self.actions[self.current_street-1])-1]
        else:
            if self.current_street == 1:
                last_action = Action()
            else:
                last_action = self.actions[self.current_street-2][len(self.actions[self.current_street-2]) - 1]
        line_2 = f"Last Action: {last_action.__repr__()} - My Bet: {self.bet2 if self.position == 0 else self.bet1} - Opp Bet: {self.bet1 if self.position == 0 else self.bet2}"
        return line_1 + line_2


@dataclass
class Action(object):
    action: constants.ACPCActions = None
    raise_amount: int = 0
    player: constants.Players = None
    street: int = 0
    index: int = 0

    def __repr__(self):
        if self.action is None:
            return "No action"
        elif self.action is constants.ACPCActions.fold:
            return "Folded"
        elif self.action is constants.ACPCActions.ccall:
            if self.raise_amount > 0:
                return f"Called {self.raise_amount}"
            else:
                return "Checked"
        elif self.action is constants.ACPCActions.rraise:
            return f"Raised {self.raise_amount}"


# --- Turns a string representation of a poker state into a table understandable by DyypHoldem.
# -- @param state a string representation of a poker state, in ACPC format
# -- @return a table of state parameters, with the fields:
# -- * `position`: which player DyypHoldem is (element of @{constants.players})
# -- * `current_street`: the current betting round
# -- * `actions`: a list of actions which reached the state, for each
# -- betting round - each action is a table with fields:
# --     * `action`: an element of @{constants.acpc_actions}
# --     * `raise_amount`: the number of chips raised (if `action` is raise)
# -- * `actions_raw`: a string representation of actions for each betting round
# -- * `all_actions`: a concatenated list of all of the actions in `actions`,
# -- with the following fields added:
# --     * `player`: the player who made the action
# --     * `street`: the betting round on which the action was taken
# --     * `index`: the index of the action in `all_actions`
# -- * `board`: a string representation of the board cards
# -- * `hand_string`: a string representation of DyypHoldem's private hand
# -- * `hand_id`: a numerical representation of DyypHoldem's private hand
# -- * `acting_player`: which player is acting (element of @{constants.players})
# -- * `bet1`, `bet2`: the number of chips committed by each player
def parse_state(state) -> ProcessedState:
    parsed_state = _parse_state(state)
    processed_state = _process_parsed_state(parsed_state)
    return processed_state


# --- Gets a representation of the public tree node which corresponds to a
# -- processed state.
# -- @param processed_state a processed state representation returned by
# -- @{parse_state}
# -- @return a table representing a public tree node, with the fields:
# -- * `street`: the current betting round
# -- * `board`: a (possibly empty) vector of board cards
# -- * `current_player`: the currently acting player
# -- * `bets`: a vector of chips committed by each player
def parsed_state_to_node(parsed_state: ProcessedState) -> TreeNode:
    node = TreeNode()
    node.street = parsed_state.current_street
    node.board = card_to_string.string_to_board(parsed_state.board)
    node.current_player = parsed_state.acting_player
    node.bets = arguments.Tensor([parsed_state.bet1, parsed_state.bet2])
    if parsed_state.bet1 != parsed_state.bet2:
        node.num_bets = 1
    return node


# --- Generates a message to send to the ACPC protocol server, given DyypHoldem's chosen action.
# -- @param last_message the last state message sent by the server
# -- @param adviced_action the action that DyypHoldem chooses to take, with fields
# -- * `action`: an element of @{constants.acpc_actions}
# -- * `raise_amount`: the number of chips to raise (if `action` is raise)
# -- @return a string messsage in ACPC format to send to the server
def action_to_message(last_message, advised_action):
    out = last_message
    protocol_action = _bet_to_protocol_action(advised_action)
    out = f"{out}:{protocol_action}"
    return out


# --- Converts an action taken by DyypHoldem into a string representation.
# -- @param adviced_action the action that DyypHoldem chooses to take, with fields
# -- * `action`: an element of @{constants.acpc_actions}
# -- * `raise_amount`: the number of chips to raise (if `action` is raise)
# -- @return a string representation of the action
# -- @local
def _bet_to_protocol_action(advised_action):
    if advised_action.action == constants.ACPCActions.ccall:
        return "c"
    elif advised_action.action == constants.ACPCActions.fold:
        return "f"
    elif advised_action.action == constants.ACPCActions.rraise:
        return f"r{advised_action.raise_amount}"
    else:
        assert False, "invalid action"


# --- Parses a set of parameters that represent a poker state, from a string
# -- representation.
# -- @param state a string representation of a poker state in ACPC format
# -- @return a table of state parameters, containing the fields:
# --
# -- * `position`: the acting player
# --
# -- * `hand_id`: a numerical id for the hand
# --
# -- * `actions`: a list of actions which reached the state, for each
# -- betting round - each action is a table with fields:
# --
# --     * `action`: an element of @{constants.acpc_actions}
# --
# --     * `raise_amount`: the number of chips raised (if `action` is raise)
# --
# -- * `actions_raw`: a string representation of actions for each betting round
# --
# -- * `board`: a string representation of the board cards
# --
# -- * `hand_p1`: a string representation of the first player's private hand
# --
# -- * `hand_p2`: a string representation of the second player's private hand
# -- @local
def _parse_state(state):

    out = ParsedState()

    (position, hand_id, actions, cards) = re.compile(r"^MATCHSTATE:(\d):(\d*):([^:]*):(.*)").search(state).groups()

    (preflop_actions, flop_actions, turn_actions, river_actions) = re.compile(r"([^/]*)/?([^/]*)/?([^/]*)/?([^/]*)").search(actions).groups()

    (hand_p1, hand_p2, flop, turn, river) = re.compile(r"([^|]*)\|([^/]*)/?([^/]*)/?([^/]*)/?([^/]*)").search(cards).groups()

    out.position = position
    out.hand_id = hand_id

    out.actions = []
    out.actions.append(_parse_actions(preflop_actions))
    out.actions.append(_parse_actions(flop_actions))
    out.actions.append(_parse_actions(turn_actions))
    out.actions.append(_parse_actions(river_actions))

    out.actions_raw = []
    out.actions_raw.append(preflop_actions)
    out.actions_raw.append(flop_actions)
    out.actions_raw.append(turn_actions)
    out.actions_raw.append(river_actions)

    out.board = f"{flop.strip()}{turn.strip()}{river.strip()}"
    out.hand_p1 = hand_p1.strip()
    out.hand_p2 = hand_p2.strip()

    return out


# --- Further processes a parsed state into a format understandable by DyypHoldem.
# -- @param parsed_state a parsed state returned by @{_parse_state}
# -- @return a table of state parameters, with the fields:
# -- * `position`: which player DyypHoldem is (element of @{constants.players})
# -- * `current_street`: the current betting round
# -- * `actions`: a list of actions which reached the state, for each
# -- betting round - each action is a table with fields:
# --     * `action`: an element of @{constants.acpc_actions}
# --     * `raise_amount`: the number of chips raised (if `action` is raise)
# -- * `actions_raw`: a string representation of actions for each betting round
# -- * `all_actions`: a concatenated list of all of the actions in `actions`,
# -- with the following fields added:
# --     * `player`: the player who made the action
# --     * `street`: the betting round on which the action was taken
# --     * `index`: the index of the action in `all_actions`
# -- * `board`: a string representation of the board cards
# -- * `hand_id`: a numerical id for the current hand
# -- * `hand_string`: a string representation of DyypHoldem's private hand
# -- * `hand_id`: a numerical representation of DyypHoldem's private hand
# -- * `acting_player`: which player is acting (element of @{constants.players})
# -- * `bet1`, `bet2`: the number of chips committed by each player
# -- @local
def _process_parsed_state(parsed_state) -> ProcessedState:

    out = ProcessedState()

    # 1.0 figure out the current street
    current_street = 1
    if parsed_state.board != '':
        current_street = int(len(parsed_state.board) / 2 - 1)

    # 2.0 convert actions to player actions
    all_actions = _convert_actions(parsed_state.actions)

    # 3.0 current board
    board = parsed_state.board

    # in protocol 0=SB 1=BB, need to convert to our representation
    out.position = int(parsed_state.position)
    out.player = constants.Players(1 - out.position)
    out.current_street = current_street
    out.actions = parsed_state.actions
    out.actions_raw = parsed_state.actions_raw
    out.all_actions = all_actions
    out.board = board
    out.hand_number = parsed_state.hand_id

    if out.position == constants.Players.P1.value:
        out.my_hand_string = parsed_state.hand_p1
        out.opponent_hand_string = parsed_state.hand_p2
    else:
        out.my_hand_string = parsed_state.hand_p2
        out.opponent_hand_string = parsed_state.hand_p1
    out.hand_id = card_tools.string_to_hole_index(out.my_hand_string)

    arguments.logger.trace(f"ACPC position: {out.position}, Player: {repr(out.player)}, Hand: {out.my_hand_string}")

    acting_player = _get_acting_player(out)
    out.acting_player = acting_player

    # 5.0 compute bets
    bets = _compute_bets(out)
    arguments.logger.trace(f"Acting Player: {repr(out.acting_player)}, Computed bets: {bets[0]}, {bets[1]}, Bets: {bets}")

    out.bet1 = bets[0]
    out.bet2 = bets[1]

    return out


# --- Parses a list of actions from a string representation.
# -- @param actions a string representing a series of actions in ACPC format
# -- @return a list of actions, each of which is a table with fields:
# -- * `action`: an element of @{constants.acpc_actions}
# -- * `raise_amount`: the number of chips raised (if `action` is raise)
# -- @local
def _parse_actions(actions: str):

    out = []
    actions_remainder: str = actions

    while actions_remainder != '':
        parsed_chunk = ''
        if actions_remainder.startswith("c"):
            out.append(Action(action=constants.ACPCActions.ccall))
            parsed_chunk = "c"
        elif actions_remainder.startswith("r"):
            raise_amount = re.compile(r"^r(\d*)").search(actions_remainder).groups()
            raise_amount = int(raise_amount[0])
            out.append(Action(action=constants.ACPCActions.rraise, raise_amount=raise_amount))
            parsed_chunk = "r" + str(raise_amount)
        elif actions_remainder.startswith("f"):
            out.append(Action(action=constants.ACPCActions.fold))
            parsed_chunk = "f"
        else:
            assert False, "unknown action"

        assert len(parsed_chunk) > 0
        actions_remainder = actions_remainder[len(parsed_chunk):]

    return out


# --- Gives the acting player at a given state.
# -- @param processed_state a table containing the fields returned by
# -- @{_process_parsed_state}, except for `acting_player`, `bet1`, and `bet2`
# -- @return the acting player, as defined by @{constants.players}
# -- @local
def _get_acting_player(processed_state: ProcessedState):
    if len(processed_state.all_actions) == 2:
        assert processed_state.current_street == 1
        return constants.Players.P1

    last_action = processed_state.all_actions[len(processed_state.all_actions) - 1]
    # has the street changed since the last action?
    if last_action.street != processed_state.current_street:
        return constants.Players.P2

    # is the hand over?
    if last_action.action == constants.ACPCActions.fold:
        return constants.Players.Chance

    if processed_state.current_street == 4 and len(processed_state.actions[3]) >= 2 and last_action.action == constants.ACPCActions.ccall:
        return constants.Players.Chance

    # there are some actions on the current street
    # the acting player is the opponent of the one who made the last action
    return constants.Players(1 - last_action.player.value)


# --- Computes the number of chips committed by each player at a state.
# -- @param processed_state a table containing the fields returned by
# -- @{_process_parsed_state}, except for `bet1` and `bet2`
# -- @return the number of chips committed by the first player
# -- @return the number of chips committed by the second player
# -- @local
def _compute_bets(processed_state):

    if processed_state.acting_player == constants.Players.Chance and processed_state.all_actions[-1].action == constants.ACPCActions.fold:
        valid_actions = len(processed_state.all_actions) - 1
    else:
        valid_actions = len(processed_state.all_actions)

    # get small blind and big blind action
    last_action = processed_state.all_actions[1]
    prev_last_action = processed_state.all_actions[0]
    prev_last_bet = last_action

    # for i in range(len(processed_state.all_actions)):
    for i in range(2, valid_actions):
        action = processed_state.all_actions[i]
        assert action.player == constants.Players.P1 or action.player == constants.Players.P2
        prev_last_action = last_action
        last_action = action
        if action.action == constants.ACPCActions.rraise and i <= (len(processed_state.all_actions) - 3):
            prev_last_bet = action

    bets = {}

    if last_action.action == constants.ACPCActions.rraise and prev_last_action.action == constants.ACPCActions.rraise:
        bets[prev_last_action.player.value] = prev_last_action.raise_amount
        bets[last_action.player.value] = last_action.raise_amount
    else:
        if last_action.action == constants.ACPCActions.ccall and prev_last_action.action == constants.ACPCActions.ccall:
            bets[0] = prev_last_bet.raise_amount
            bets[1] = prev_last_bet.raise_amount
        else:
            # either ccal/raise or raise/ccal situation
            # raise/ccall
            if last_action.action == constants.ACPCActions.ccall:
                assert prev_last_action.action == constants.ACPCActions.rraise and prev_last_action.raise_amount
                bets[0] = prev_last_action.raise_amount
                bets[1] = prev_last_action.raise_amount
            else:
                # call/raise
                assert last_action.action == constants.ACPCActions.rraise and last_action.raise_amount
                bets[last_action.player.value] = last_action.raise_amount
                bets[1 - last_action.player.value] = prev_last_bet.raise_amount

    return bets


# --- Processes all actions.
# -- @param actions a list of actions for each betting round
# -- @return a of list actions, processed with @{_convert_actions_street} and
# -- concatenated
# -- @local
def _convert_actions(actions):
    all_actions = []
    for street in range(4):
        _convert_actions_street(actions[street], street+1, all_actions)
    return all_actions


# --- Processes a list of actions for a betting round.
# -- @param actions a list of actions (see @{_parse_actions})
# -- @param street the betting round on which the actions takes place
# -- @param all_actions A list which the actions are appended to. Fields `player`,
# -- `street`, and `index` are added to each action.
# -- @local
def _convert_actions_street(actions, street, all_actions):

    street_first_player = street == 1 and constants.Players.P1 or constants.Players.P2

    if street == 1:
        first_p1_action = Action(action=constants.ACPCActions.rraise, raise_amount=game_settings.small_blind,
                                 player=constants.Players.P1, street=1)
        first_p2_action = Action(action=constants.ACPCActions.rraise, raise_amount=game_settings.big_blind,
                                 player=constants.Players.P2, street=1)
        all_actions.append(first_p1_action)
        all_actions.append(first_p2_action)

    for i in range(len(actions)):
        acting_player = -1
        if i % 2 == 0:
            acting_player = street_first_player
        else:
            acting_player = constants.Players(1 - street_first_player.value)
        action = actions[i]
        action.player = acting_player
        action.street = street
        action.index = len(all_actions) + 1
        all_actions.append(action)
