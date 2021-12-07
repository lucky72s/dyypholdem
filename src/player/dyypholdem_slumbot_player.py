import os
import sys
import argparse
sys.path.append(os.getcwd())


def play_hand(token, hand):

    winnings = 0

    response = slumbot_game.new_hand(token)
    new_token = response.get('token')
    if new_token:
        token = new_token
    arguments.logger.trace(f"Current token: {token}")

    current_state, current_node = slumbot_game.get_next_situation(response)

    winnings = response.get('winnings')
    # game goes on
    if winnings is None:

        arguments.logger.info(f"Starting new hand #{hand+1}")
        continual_resolving.start_new_hand(current_state)

        while True:
            # use continual resolving to find a strategy and make an action in the current node
            advised_action: protocol_to_node.Action = continual_resolving.compute_action(current_state, current_node)

            # send the action to the server
            response = slumbot_game.play_action(token, advised_action)
            current_state, current_node = slumbot_game.get_next_situation(response)

            winnings = response.get('winnings')
            if winnings is not None:
                # hand has ended
                break

    # clean up and release memory
    if arguments.use_gpu:
        arguments.logger.trace(f"Initiating garbage collection. Allocated memory={torch.cuda.memory_allocated('cuda')}, Reserved memory={torch.cuda.memory_reserved('cuda')}")
    del current_node
    del current_state
    gc.collect()
    if arguments.use_gpu:
        torch.cuda.empty_cache()
        arguments.logger.trace(f"Garbage collection performed. Allocated memory={torch.cuda.memory_allocated('cuda')}, Reserved memory={torch.cuda.memory_reserved('cuda')}")

    return token, winnings


def play_slumbot():
    token = None
    num_hands = args.hands
    winnings = 0
    for hand in range(num_hands):
        token, hand_winnings = play_hand(token, hand)
        winnings += hand_winnings
        arguments.logger.success(f"Hand completed. Hand winnings: {hand_winnings}, Total winnings: {winnings} ")

    arguments.logger.success(f"Game ended >>> Total winnings: {winnings}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Play with DyypHoldem against Slumbot')
    parser.add_argument('hands', type=int, help="Number of hands to play against Slumbot")
    args = parser.parse_args()

    import gc

    import torch

    import settings.arguments as arguments

    from server.slumbot_game import SlumbotGame
    import server.protocol_to_node as protocol_to_node
    from lookahead.continual_resolving import ContinualResolving

    slumbot_game = SlumbotGame()
    continual_resolving = ContinualResolving()

    play_slumbot()
