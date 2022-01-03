import requests
import sys

import settings.arguments as arguments
import settings.constants as constants
import settings.game_settings as game_settings

import server.protocol_to_node as protocol_to_node

host = 'slumbot.com'

NUM_STREETS = constants.streets_count
SMALL_BLIND = 50
BIG_BLIND = 100
STACK_SIZE = 20000
game_settings.small_blind = SMALL_BLIND
game_settings.big_blind = BIG_BLIND
game_settings.stack = STACK_SIZE


class SlumbotGame(object):
    last_response: dict
    acpc_actions: str
    current_state: dict
    max_bet: int
    bet_this_street: int
    bet_previous_streets: int
    current_street: int

    def __init__(self):
        pass

    def new_hand(self, token):
        self.last_response = None
        self.acpc_actions = ""
        self.max_bet = 0
        self.bet_this_street = 0
        self.bet_previous_streets = 0
        self.current_street = 0

        data = {}
        if token:
            data['token'] = token
        # Use verify=false to avoid SSL Error
        response = requests.post(f'https://{host}/api/new_hand', headers={}, json=data)
        success = getattr(response, 'status_code') == 200
        if not success:
            arguments.logger.critical('Status code: %s' % repr(response.status_code))
            try:
                arguments.logger.error('Error response: %s' % repr(response.json()))
            except ValueError:
                pass
            sys.exit(-1)

        try:
            r = response.json()
        except ValueError:
            arguments.logger.critical('Could not get JSON from response')
            sys.exit(-1)

        if 'error_msg' in r:
            arguments.logger.critical('Error: %s' % r['error_msg'])
            sys.exit(-1)

        return r

    def get_next_situation(self, response):
        arguments.logger.trace(f"Message from server: {repr(response)}")
        action = response.get('action')
        self.current_state = self.parse_action(action)
        if 'error' in self.current_state:
            arguments.logger.critical('Error parsing action %s: %s' % (action, self.current_state['error']))
            sys.exit(-1)
        if self.current_street != self.current_state['st']:
            self.current_street = self.current_state['st']
            self.bet_previous_streets += self.bet_this_street
            self.bet_this_street = 0

        msg = self.convert_state(response, self.current_state)
        arguments.logger.info(f"New state received from server: {msg}")
        parsed_state = protocol_to_node.parse_state(msg)
        node = protocol_to_node.parsed_state_to_node(parsed_state)

        self.last_response = response

        return parsed_state, node

    def convert_state(self, response, state):
        prefix = "MATCHSTATE"
        position = response.get('client_pos')
        hole_cards_list = response.get('hole_cards')
        board_cards_list = response.get('board')
        hole_cards = ""
        for i in range(len(hole_cards_list)):
            hole_cards += hole_cards_list[i]
        if position == 0:
            hole_cards += "|"
        else:
            hole_cards = "|" + hole_cards
        board_cards = ""
        for i in range(len(board_cards_list)):
            if i == 0 or i == 3 or i == 4:
                board_cards += "/"
            board_cards += board_cards_list[i]
        self.acpc_actions, self.max_bet = self.acpcify_actions(response.get('action'))
        return f"{prefix}:{position}:0:{self.acpc_actions}:{hole_cards}{board_cards}"

    def play_action(self, token, advised_action: protocol_to_node.Action):
        next_action = ""
        if advised_action.action == constants.ACPCActions.fold:
            next_action = "f"
        elif advised_action.action == constants.ACPCActions.ccall:
            if self.current_state["street_last_bet_to"] == 0:
                next_action = "k"
            else:
                next_action = "c"
                arguments.logger.debug(f"Calling bet of: {self.current_state['street_last_bet_to']}")
                self.bet_this_street += self.current_state["street_last_bet_to"]
        elif advised_action.action == constants.ACPCActions.rraise:
            raise_amount = advised_action.raise_amount
            arguments.logger.trace(f"Raise amount: {raise_amount}")
            self.bet_this_street = raise_amount - self.bet_previous_streets
            arguments.logger.trace(f"Calculated bet size: {self.bet_this_street}")
            if self.bet_this_street + self.bet_previous_streets > game_settings.stack:
                self.bet_this_street = game_settings.stack - self.bet_previous_streets
                arguments.logger.warning(f"Bet size corrected for all-in: {self.bet_this_street}")
            elif self.bet_this_street < self.max_bet:
                self.bet_this_street = self.max_bet
                arguments.logger.warning(f"Bet size corrected to be min bet size: {self.bet_this_street}")
            next_action = f"b{self.bet_this_street}"

        arguments.logger.debug(f"Sending action to server: {next_action}")

        data = {'token': token, 'incr': next_action}
        # Use verify=false to avoid SSL Error
        response = requests.post(f'https://{host}/api/act', headers={}, json=data)
        success = getattr(response, 'status_code') == 200
        if not success:
            arguments.logger.critical('Status code: %s' % repr(response.status_code))
            try:
                arguments.logger.error('Error response: %s' % repr(response.json()))
            except ValueError:
                pass
            sys.exit(-1)
        try:
            r = response.json()
        except ValueError:
            arguments.logger.critical('Could not get JSON from response')
            sys.exit(-1)
        if 'error_msg' in r:
            arguments.logger.error('Error: %s' % r['error_msg'])
            sys.exit(-1)

        return r

    @staticmethod
    def login(username, password):
        data = {"username": username, "password": password}
        # If porting this code to another language, make sure that the Content-Type header is
        # set to application/json.
        response = requests.post(f'https://{host}/api/login', json=data)
        success = getattr(response, 'status_code') == 200
        if not success:
            print('Status code: %s' % repr(response.status_code))
            try:
                print('Error response: %s' % repr(response.json()))
            except ValueError:
                pass
            sys.exit(-1)

        try:
            r = response.json()
        except ValueError:
            print('Could not get JSON from response')
            sys.exit(-1)

        if 'error_msg' in r:
            print('Error: %s' % r['error_msg'])
            sys.exit(-1)
            
        token = r.get('token')
        if not token:
            print('Did not get token in response to /api/login')
            sys.exit(-1)
        return token

    @staticmethod
    def acpcify_actions(actions):
        actions = actions.replace("b", "r")
        actions = actions.replace("k", "c")
        streets = actions.split("/")
        max_bet = 0
        for i, street_actions in enumerate(streets):
            bets = street_actions.split("r")
            max_street_bet = max_bet
            for j, betstr in enumerate(bets):
                try:
                    flag_c = False
                    flag_f = False
                    if len(betstr) > 1 and betstr[-1] == 'c':
                        flag_c = True
                        betstr = betstr.replace("c", "")
                    elif len(betstr) > 1 and betstr[-1] == 'f':
                        flag_f = True
                        betstr = betstr.replace("f", "")
                    bet = int(betstr)
                    bet += max_bet
                    max_street_bet = max(max_street_bet, bet)
                    bets[j] = str(bet)
                    if flag_c:
                        bets[j] += "c"
                    elif flag_f:
                        bets[j] += "f"
                    bets[j] = "r" + bets[j]
                except ValueError:
                    continue
            max_bet = max_street_bet
            if max_bet == 0:
                max_bet = 100
            good_string = "".join(bets)
            streets[i] = good_string
        return "/".join(streets), max_bet

    @staticmethod
    def parse_action(action):
        """
            Returns a dict with information about the action passed in.
            Returns a key "error" if there was a problem parsing the action.
            pos is returned as -1 if the hand is over; otherwise the position of the player next to act.
            street_last_bet_to only counts chips bet on this street, total_last_bet_to counts all
              chips put into the pot.
            Handles action with or without a final '/'; e.g., "ck" or "ck/".
            """
        st = 0
        street_last_bet_to = BIG_BLIND
        total_last_bet_to = BIG_BLIND
        last_bet_size = BIG_BLIND - SMALL_BLIND
        last_bettor = 0
        sz = len(action)
        pos = 1
        if sz == 0:
            return {
                'st': st,
                'pos': pos,
                'street_last_bet_to': street_last_bet_to,
                'total_last_bet_to': total_last_bet_to,
                'last_bet_size': last_bet_size,
                'last_bettor': last_bettor,
            }

        check_or_call_ends_street = False
        i = 0
        while i < sz:
            if st >= NUM_STREETS:
                return {'error': 'Unexpected error'}
            c = action[i]
            i += 1
            if c == 'k':
                if last_bet_size > 0:
                    return {'error': 'Illegal check'}
                if check_or_call_ends_street:
                    # After a check that ends a pre-river street, expect either a '/' or end of string.
                    if st < NUM_STREETS - 1 and i < sz:
                        if action[i] != '/':
                            return {'error': 'Missing slash'}
                        i += 1
                    if st == NUM_STREETS - 1:
                        # Reached showdown
                        pos = -1
                    else:
                        pos = 0
                        st += 1
                    street_last_bet_to = 0
                    check_or_call_ends_street = False
                else:
                    pos = (pos + 1) % 2
                    check_or_call_ends_street = True
            elif c == 'c':
                if last_bet_size == 0:
                    return {'error': 'Illegal call'}
                if total_last_bet_to == STACK_SIZE:
                    # Call of an all-in bet
                    # Either allow no slashes, or slashes terminating all streets prior to the river.
                    if i != sz:
                        for st1 in range(st, NUM_STREETS - 1):
                            if i == sz:
                                return {'error': 'Missing slash (end of string)'}
                            else:
                                c = action[i]
                                i += 1
                                if c != '/':
                                    return {'error': 'Missing slash'}
                    if i != sz:
                        return {'error': 'Extra characters at end of action'}
                    st = NUM_STREETS - 1
                    pos = -1
                    last_bet_size = 0
                    return {
                        'st': st,
                        'pos': pos,
                        'street_last_bet_to': street_last_bet_to,
                        'total_last_bet_to': total_last_bet_to,
                        'last_bet_size': last_bet_size,
                        'last_bettor': last_bettor,
                    }
                if check_or_call_ends_street:
                    # After a call that ends a pre-river street, expect either a '/' or end of string.
                    if st < NUM_STREETS - 1 and i < sz:
                        if action[i] != '/':
                            return {'error': 'Missing slash'}
                        i += 1
                    if st == NUM_STREETS - 1:
                        # Reached showdown
                        pos = -1
                    else:
                        pos = 0
                        st += 1
                    street_last_bet_to = 0
                    check_or_call_ends_street = False
                else:
                    pos = (pos + 1) % 2
                    check_or_call_ends_street = True
                last_bet_size = 0
                last_bettor = -1
            elif c == 'f':
                if last_bet_size == 0:
                    return {'error', 'Illegal fold'}
                if i != sz:
                    return {'error': 'Extra characters at end of action'}
                pos = -1
                return {
                    'st': st,
                    'pos': pos,
                    'street_last_bet_to': street_last_bet_to,
                    'total_last_bet_to': total_last_bet_to,
                    'last_bet_size': last_bet_size,
                    'last_bettor': last_bettor,
                }
            elif c == 'b':
                j = i
                while i < sz and action[i] >= '0' and action[i] <= '9':
                    i += 1
                if i == j:
                    return {'error': 'Missing bet size'}
                try:
                    new_street_last_bet_to = int(action[j:i])
                except (TypeError, ValueError):
                    return {'error': 'Bet size not an integer'}
                new_last_bet_size = new_street_last_bet_to - street_last_bet_to
                # Validate that the bet is legal
                remaining = STACK_SIZE - street_last_bet_to
                if last_bet_size > 0:
                    min_bet_size = last_bet_size
                    # Make sure minimum opening bet is the size of the big blind.
                    if min_bet_size < BIG_BLIND:
                        min_bet_size = BIG_BLIND
                else:
                    min_bet_size = BIG_BLIND
                # Can always go all-in
                if min_bet_size > remaining:
                    min_bet_size = remaining
                if new_last_bet_size < min_bet_size:
                    return {'error': f"Bet too small - remaining={remaining}, min_bet_size={min_bet_size}, new_last_bet_size={new_last_bet_size}"}
                max_bet_size = remaining
                if new_last_bet_size > max_bet_size:
                    return {'error': f"Bet too big - remaining={remaining}, max_bet_size={max_bet_size}, new_last_bet_size={new_last_bet_size}"}
                last_bet_size = new_last_bet_size
                street_last_bet_to = new_street_last_bet_to
                total_last_bet_to += last_bet_size
                last_bettor = pos
                pos = (pos + 1) % 2
                check_or_call_ends_street = True
            else:
                return {'error': 'Unexpected character in action'}

        return {
            'st': st,
            'pos': pos,
            'street_last_bet_to': street_last_bet_to,
            'total_last_bet_to': total_last_bet_to,
            'last_bet_size': last_bet_size,
            'last_bettor': last_bettor,
        }


