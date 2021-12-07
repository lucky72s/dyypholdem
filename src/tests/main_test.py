import os
import sys
import argparse
sys.path.append(os.getcwd())


def run_test(node, player_range, opponent_range):
    te = TerminalEquity()
    te.set_board(current_node.board)
    resolving = Resolving(te)
    # calculate results
    return resolving.resolve_first_node(node, player_range, opponent_range)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run test of a specific poker hand on the defined street')
    parser.add_argument('street', type=int, choices=[1, 2, 3, 4], help="Street (1=pre-flop, 2=flop, 3=turn, 4=river)")
    args = parser.parse_args()

    import settings.arguments as arguments
    import test_river
    import test_turn
    import test_flop
    import test_preflop

    from terminal_equity.terminal_equity import TerminalEquity
    from lookahead.resolving import Resolving
    import utils.output as output

    street = args.street
    arguments.logger.info(f"Running test for street: {arguments.street_names[street]}")
    prepare_test = None
    if street == 4:
        prepare_test = test_river.prepare_test
    elif street == 3:
        prepare_test = test_turn.prepare_test
    elif street == 2:
        prepare_test = test_flop.prepare_test
    elif street == 1:
        prepare_test = test_preflop.prepare_test

    arguments.timer.start()

    # prepare test
    current_node, player_range_tensor, opponent_range_tensor = prepare_test()
    # calculate results
    results = run_test(current_node, player_range_tensor, opponent_range_tensor)
    # output results
    output.show_results(player_range_tensor, current_node, results)

    arguments.timer.stop("Testing completed in: ", log_level="TIMING")
    arguments.logger.success("Test completed.")
