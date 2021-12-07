
import settings.arguments as arguments
import settings.game_settings as game_settings

import game.card_tools as card_tools
import game.card_to_string_conversion as card_to_string


class DummyLogger(object):

    def __init__(self, log_level):
        self.log_level = self._level_as_number(log_level)

    @staticmethod
    def _level_as_number(level) -> int:
        if level == 'TRACE':
            return 5
        elif level == 'LOADING':
            return 8
        elif level == 'DEBUG':
            return 10
        elif level == 'TIMING':
            return 15
        elif level == 'INFO':
            return 20
        elif level == 'SUCCESS':
            return 25
        elif level == 'WARNING':
            return 30
        elif level == 'ERROR':
            return 40
        elif level == 'CRITICAL':
            return 50

    def log(self, level, msg):
        _level = self._level_as_number(level)
        if _level >= self.log_level:
            if level == 'TRACE':
                print(f"TRACE:     {msg}")
            elif level == 'LOADING':
                print(f"LOADING:   {msg}")
            elif level == 'DEBUG':
                print(f"DEBUG:     {msg}")
            elif level == 'TIMING':
                print(f"TIMING:    {msg}")
            elif level == 'INFO':
                print(f"INFO:      {msg}")
            elif level == 'SUCCESS':
                print(f"SUCCESS:   {msg}")
            elif level == 'WARNING':
                print(f"WARNING:   {msg}")
            elif level == 'ERROR':
                print(f"ERROR:     {msg}")
            elif level == 'CRITICAL':
                print(f"CRITICAL:  {msg}")

    def trace(self, msg):
        self.log('TRACE', msg)

    def loading(self, msg):
        self.log('LOADING', msg)

    def debug(self, msg):
        self.log('DEBUG', msg)

    def timing(self, msg):
        self.log('TIMING', msg)

    def info(self, msg):
        self.log('INFO', msg)

    def success(self, msg):
        self.log('SUCCESS', msg)

    def warning(self, msg):
        self.log('WARNING', msg)

    def error(self, msg):
        self.log('ERROR', msg)

    def critical(self, msg):
        self.log('CRITICAL', msg)


def show_results(player_range, node, results):
    for card1 in range(game_settings.card_count):
        for card2 in range(card1 + 1, game_settings.card_count):
            idx = card_tools.get_hole_index([card1, card2])
            if player_range[0][idx] > 0:
                result = f"{card_to_string.card_to_string(card1)}{card_to_string.card_to_string(card2)}: {'{:.3f}'.format(results.root_cfvs_both_players[0][0][idx])}"
                for i in range(results.strategy.size(0)):
                    action = int(results.get_actions()[i])
                    if action == -2:
                        action_str = "   \tFold"
                    elif action == -1:
                        if node.bets[0] == node.bets[1]:
                            action_str = "Check"
                        else:
                            action_str = "Call"
                    elif action == game_settings.stack:
                        action_str = "All-In"
                    else:
                        action_str = f"Bet {action}"
                    result += f"      {action_str}: {results.strategy[i][0][idx]:.6f}"
                arguments.logger.info(result)
