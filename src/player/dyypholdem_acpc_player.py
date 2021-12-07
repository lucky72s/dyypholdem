import os
import sys
import argparse
sys.path.append(os.getcwd())


last_state = None
last_node = None


def run(server, port):
    global last_state
    global last_node

    # 1.0 connecting to the server
    acpc_game = ACPCGame()
    acpc_game.connect(server, port)

    current_state: protocol_to_node.ProcessedState
    current_node: TreeNode

    winnings = 0

    # 2.0 main loop that waits for a situation where we act and then chooses an action
    while True:

        # 2.1 blocks until it's our situation/turn
        current_state, current_node, hand_winnings = acpc_game.get_next_situation()

        if current_state is None:
            # game ended or connection to server broke
            break

        if current_node is not None:
            # do we have a new hand?
            if last_state is None or last_state.hand_number != current_state.hand_number or current_node.street < last_node.street:
                arguments.logger.trace(
                    f"Initiating garbage collection. Allocated memory={torch.cuda.memory_allocated('cuda')}, Reserved memory={torch.cuda.memory_reserved('cuda')}")
                del last_node
                del last_state
                gc.collect()
                if arguments.use_gpu:
                    torch.cuda.empty_cache()
                    arguments.logger.trace(
                        f"Garbage collection completed. Allocated memory={torch.cuda.memory_allocated('cuda')}, Reserved memory={torch.cuda.memory_reserved('cuda')}")
                continual_resolving.start_new_hand(current_state)

            # 2.1 use continual resolving to find a strategy and make an action in the current node
            advised_action: protocol_to_node.Action = continual_resolving.compute_action(current_state, current_node)

            if advised_action.action == constants.ACPCActions.ccall:
                advised_action.raise_amount = abs(current_state.bet1 - current_state.bet2)

            # 2.2 send the action to the dealer
            acpc_game.play_action(advised_action)

            last_state = current_state
            last_node = current_node

            # force clean up
            arguments.logger.trace(
                f"Initiating garbage collection. Allocated memory={torch.cuda.memory_allocated('cuda')}, Reserved memory={torch.cuda.memory_reserved('cuda')}")
            gc.collect()
            if arguments.use_gpu:
                torch.cuda.empty_cache()
                arguments.logger.trace(
                    f"Garbage collection completed. Allocated memory={torch.cuda.memory_allocated('cuda')}, Reserved memory={torch.cuda.memory_reserved('cuda')}")
        else:
            winnings += hand_winnings
            arguments.logger.success(f"Hand completed. Hand winnings: {hand_winnings}, Total winnings: {winnings}")

    arguments.logger.success(f"Game ended >>> Total winnings: {winnings}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Play poker on an ACPC server')
    parser.add_argument('hostname', type=str, help="Hostname/IP of the server running ACPC dealer")
    parser.add_argument('port', type=int, help="Port to connect on the ACPC server")
    args = parser.parse_args()

    import gc

    import torch

    import settings.arguments as arguments
    import settings.constants as constants

    from server.acpc_game import ACPCGame
    import server.protocol_to_node as protocol_to_node
    from tree.tree_node import TreeNode
    from lookahead.continual_resolving import ContinualResolving

    import utils.pseudo_random as random_

    continual_resolving = ContinualResolving()

    if arguments.use_pseudo_random:
        random_.manual_seed(0)

    run(args.hostname, args.port)
