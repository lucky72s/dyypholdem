
import settings.arguments as arguments
import settings.constants as constants
import settings.game_settings as game_settings

from server.network_communication import ACPCNetworkCommunication
import server.protocol_to_node as protocol_to_node

from tree.tree_node import TreeNode
from game.evaluation.evaluator import Evaluator
import game.card_to_string_conversion as card_conversion


class ACPCGame(object):

    debug_msg: str
    network_communication: ACPCNetworkCommunication
    last_msg: str

    def __init__(self):
        pass

    # --- Connects to a specified ACPC server which acts as the dealer.
    # --
    # -- @param server the server that sends states to DyypHoldem, which responds with actions
    # -- @param port the port to connect on
    # -- @see network_communication.connect
    def connect(self, server, port):
        arguments.logger.debug(f"Connecting to ACPC server with IP={server} and port={port}")
        self.network_communication = ACPCNetworkCommunication()
        self.network_communication.connect(server, port)

    def string_to_state_node(self, msg):
        arguments.logger.trace(f"Parsing new state from server: {msg}")
        parsed_state = protocol_to_node.parse_state(msg)
        # current player to act is us
        if parsed_state.acting_player == parsed_state.position:
            # we should not act since this is an allin situations
            if parsed_state.bet1 == parsed_state.bet2 and parsed_state.bet1 == game_settings.stack:
                arguments.logger.debug("Not our turn -or- all in")
            # we should act
            else:
                arguments.logger.debug("Our turn >>>")
                self.last_msg = msg
                # create a tree node from the current state
                node = protocol_to_node.parsed_state_to_node(parsed_state)
                return parsed_state, node
        # current player to act is the opponent
        else:
            arguments.logger.debug("Not our turn...")
        return None, None

    # --- Receives and parses the next poker situation where DyypHoldem must act.
    # --
    # -- Blocks until the server sends a situation where DyypHoldem acts.
    # -- @return the parsed state representation of the poker situation (see
    # -- @{protocol_to_node.parse_state})
    # -- @return a public tree node for the state (see
    # -- @{protocol_to_node.parsed_state_to_node})
    def get_next_situation(self) -> (protocol_to_node.ProcessedState, TreeNode):
        while True:
            msg = None

            # 1.0 get the message from the dealer
            msg = self.network_communication.get_line()

            if not msg:
                arguments.logger.trace("Received empty message from server -> ending game")
                return None, None, 0

            arguments.logger.info(f"Received ACPC dealer message: {msg.strip()}")

            # 2.0 parse the string to our state representation
            parsed_state = protocol_to_node.parse_state(msg)
            arguments.logger.debug(parsed_state)

            # 3.0 figure out if we should act
            # current player to act is us
            if parsed_state.acting_player == constants.Players.Chance:
                # hand has ended
                my_bet = parsed_state.bet2 if parsed_state.position == 0 else parsed_state.bet1
                opp_bet = parsed_state.bet1 if parsed_state.position == 0 else parsed_state.bet2
                if parsed_state.all_actions[-1].action is constants.ACPCActions.fold:
                    have_won = my_bet >= opp_bet
                else:
                    my_final_hand = parsed_state.my_hand_string + parsed_state.board
                    opp_final_hand = parsed_state.opponent_hand_string + parsed_state.board
                    my_strength = Evaluator.evaluate_seven_card_hand(card_conversion.string_to_board(my_final_hand))
                    opp_strength = Evaluator.evaluate_seven_card_hand(card_conversion.string_to_board(opp_final_hand))
                    have_won = my_strength.item() <= opp_strength.item()

                winner = parsed_state.player if have_won else (constants.Players(1 - parsed_state.player.value))
                winnings = opp_bet if have_won else -my_bet
                arguments.logger.trace(f"Hand ended with winner {winner}")
                arguments.logger.trace(f"Final bets: {parsed_state.player}={my_bet}, {constants.Players(1 - parsed_state.player.value)}={opp_bet}")

                return parsed_state, None, winnings

            elif parsed_state.acting_player == parsed_state.player:
                # we should not act since this is an allin situations
                if parsed_state.bet1 == parsed_state.bet2 and parsed_state.bet1 == game_settings.stack:
                    arguments.logger.debug("All in situation")
                # we should act
                else:
                    arguments.logger.debug("Our turn >>>")
                    self.last_msg = msg.strip()
                    # create a tree node from the current state
                    node = protocol_to_node.parsed_state_to_node(parsed_state)
                    return parsed_state, node, 0
            # current player to act is the opponent
            else:
                arguments.logger.debug("Not our turn...")

    # --- Informs the server that DyypHoldem is playing a specified action.
    # -- @param adviced_action a table specifying the action chosen by DyypHoldem, with the fields:
    # -- * `action`: an element of @{constants.acpc_actions}
    # -- * `raise_amount`: the number of chips raised (if `action` is raise)
    def play_action(self, advised_action):
        message = protocol_to_node.action_to_message(self.last_msg, advised_action)
        arguments.logger.debug(f"Sending action message to the ACPC dealer: {message}")
        self.network_communication.send_line(message)
