import os
import sys
sys.path.append(os.getcwd())


last_state = None
last_node = None

# game_messages = ["MATCHSTATE:0:0:r300:Qs9d|", "MATCHSTATE:0:0:r300c/:Qs9d|/6d4d3c"]
game_messages = ["MATCHSTATE:1:3::|TcAd", "MATCHSTATE:1:3:r300c/c:|TcAd/Ts8c6h", "MATCHSTATE:1:3:r300c/cc/r900:|TcAd/Ts8c6h/As"]
# game_messages = ["MATCHSTATE:0:0:r200:Ad9h|", "MATCHSTATE:0:0:r200c/:Ad9h|/Ac9s9d", "MATCHSTATE:0:0:r200c/cc/:Ad9h|/Ac9s9d/6s"]


def replay():
    for message in game_messages:
        run(message)


def run(msg):
    global last_state
    global last_node

    # parse the state message
    current_state, current_node = get_state(msg)

    # do we have a new hand?
    if last_state is None or last_state.hand_number != current_state.hand_number or current_node.street < last_node.street:
        arguments.logger.info("Starting new hand")
        del last_state
        del last_node
        # force clean up
        arguments.logger.trace(
            f"Initiating garbage collection. Allocated memory={torch.cuda.memory_allocated('cuda')}, Reserved memory={torch.cuda.memory_reserved('cuda')}")
        gc.collect()
        if arguments.use_gpu:
            torch.cuda.empty_cache()
            arguments.logger.trace(
                f"Garbage collection performed. Allocated memory={torch.cuda.memory_allocated('cuda')}, Reserved memory={torch.cuda.memory_reserved('cuda')}")
        continual_resolving.start_new_hand(current_state)

    # use continual resolving to find a strategy and make an action in the current node
    advised_action: protocol_to_node.Action = continual_resolving.compute_action(current_state, current_node)

    last_state = current_state
    last_node = current_node

    # force clean up
    if arguments.use_gpu:
        arguments.logger.trace(
            f"Initiating garbage collection. Allocated memory={torch.cuda.memory_allocated('cuda')}, Reserved memory={torch.cuda.memory_reserved('cuda')}")
    gc.collect()
    if arguments.use_gpu:
        torch.cuda.empty_cache()
        arguments.logger.trace(f"Garbage collection performed. Allocated memory={torch.cuda.memory_allocated('cuda')}, Reserved memory={torch.cuda.memory_reserved('cuda')}")


def get_state(msg):
    arguments.logger.info(f"Parsing new state: {msg}")
    # parse the string to our state representation
    parsed_state = protocol_to_node.parse_state(msg)

    # figure out if we should act
    # current player to act is us
    if parsed_state.acting_player == parsed_state.player:
        # we should not act since this is an allin situations
        if parsed_state.bet1 == parsed_state.bet2 and parsed_state.bet1 == game_settings.stack:
            arguments.logger.debug("State parsed -> not our turn -or- all in")
        # we should act
        else:
            arguments.logger.debug("State parsed -> our turn >>>")
            # create a tree node from the current state
            node = protocol_to_node.parsed_state_to_node(parsed_state)
            return parsed_state, node
    # current player to act is the opponent
    else:
        arguments.logger.debug("State parsed -> not our turn...")


if __name__ == "__main__":
    import gc

    import torch

    import settings.arguments as arguments
    import settings.game_settings as game_settings

    import server.protocol_to_node as protocol_to_node
    from lookahead.continual_resolving import ContinualResolving

    import utils.pseudo_random as random_

    continual_resolving = ContinualResolving()

    arguments.logger.info("Running test")
    random_.manual_seed(0)
    replay()
    arguments.logger.success("Test completed")
