import os
import sys
import argparse
sys.path.append(os.getcwd())


def run(server, port):

    # 1.0 connecting to the server
    acpc_game = ACPCGame()
    acpc_game.connect(server, port)

    state: ProcessedState
    winnings = 0

    # 2.0 main loop that waits for a situation where we act and then chooses an action
    while True:
        # 2.1 blocks until it's our situation/turn
        state, node, hand_winnings = acpc_game.get_next_situation()

        if state is None:
            # game ended or connection to server broke
            break

        print(Fore.WHITE + Style.BRIGHT + f"\nNew state >>> {repr(state)}")

        if node is not None:
            # 2.2 get the player's action
            # print("Please enter your action (f/c/#):")
            action = input(Fore.LIGHTBLUE_EX + Style.BRIGHT + "Please enter your next action (f/c/#): ")

            if action == "f":
                acpc_action = Action(action=constants.ACPCActions.fold)
            elif action == "c":
                acpc_action = Action(action=constants.ACPCActions.ccall, raise_amount=abs(state.bet1 - state.bet2))
            else:
                amount = int(action)
                acpc_action = Action(action=constants.ACPCActions.rraise, raise_amount=amount)

            # 2.3 send the action to the dealer
            acpc_game.play_action(acpc_action)
        else:
            # hand has ended
            winnings += hand_winnings
            arguments.logger.success(f"Hand completed. Hand winnings: {hand_winnings}, Total winnings: {winnings}")
            print(Fore.GREEN + Style.BRIGHT + f"Hand completed. Hand winnings: {hand_winnings}, Total winnings: {winnings}")

    arguments.logger.success(f"Game ended >>> Total winnings: {winnings}")
    print(Fore.GREEN + Style.BRIGHT + f"Game ended >>> Total winnings: {winnings}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Play poker on an ACPC server')
    parser.add_argument('hostname', type=str, help="Hostname/IP of the server running ACPC dealer")
    parser.add_argument('port', type=int, help="Port to connect on the ACPC server")
    args = parser.parse_args()

    from colorama import Fore, Style

    import settings.arguments as arguments
    import settings.constants as constants

    from server.acpc_game import ACPCGame
    from server.protocol_to_node import Action, ProcessedState

    arguments.logger.remove(1)

    run(args.hostname, args.port)
