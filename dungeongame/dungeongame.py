from dungeongame.player import Player
from dungeongame.deck import Deck
from dungeongame.equipment import Equipment
from dungeongame.card import Card
from dungeongame.adventurer import Adventurer
from dungeongame.util import ask_for_input

import random
from numpy import roll
import uuid

# Constant to populate the initial monster deck
INITIAL_MONSTER_DECK = [
    ("Goblin", 1, [Equipment.TORCH]),
    ("Goblin", 1, [Equipment.TORCH]),
    ("Skeleton", 2, [Equipment.TORCH, Equipment.CHALICE]),
    ("Skeleton", 2, [Equipment.TORCH, Equipment.CHALICE]),
    ("Orc", 3, [Equipment.TORCH]),
    ("Orc", 3, [Equipment.TORCH]),
    ("Vampire", 4, [Equipment.CHALICE]),
    ("Vampire", 4, [Equipment.CHALICE]),
    ("Golem", 5, []),
    ("Golem", 5, []),
    ("Wraith", 6, [Equipment.CHALICE]),
    ("Demon", 7, []),
    ("Dragon", 9, [Equipment.LANCE]),
]

# Random names for the players
NAMES = [
    "Alice",
    "Bob",
    "Charlie",
    "David",
    "Eve",
    "Frank",
    "Grace",
    "Hannah",
    "Ivan",
    "Julia",
    "Kevin",
    "Linda",
]
random.shuffle(NAMES)

# Constant to store the results of a turn
TURN_RESULTS = {
    "passed": None,
    "equipment_taken": None,
    "monster_added": None,
    "dungeon_result": None,
}

REWARDS = {
    "died": -100,
    "lost": -200,
    "passed": 100,
    "won": 200
}


class DungeonGame:
    """ This class will store the game variables and run the game loop."""
    def __init__(self, players, model_path=None, agent=None, frozen_agent=None, auto_run: bool = False,
                 log_game: bool = False, verbose: bool = False):
        """
        Game constructor

        Parameters
        ----------
        players : list or int
            A list of players to play the game. Each player can be a dictionary
            with the following keys
            - name: The name of the player
            - mode: The mode of the player. Can be "random", "greedy" or "human"
            - action_selection_mode: The mode of action selection.
                                     Can be "random_dist" (default) or "greedy"
            If player is an integer, the game will create that many random players
        auto_run : bool
            If True, the game will run automatically without asking for user input
        log_game : bool
            If True, the game will log the game state and actions taken by the players
        verbose : bool
            If True, the game will print the game state and actions taken by the players
        """
        
        self.model_path = model_path
        
        if type(players) == int:
            if players < 2:
                raise ValueError("The game must have at least 2 players!")
            self.players = [Player(NAMES[idx], mode="random", model_path=self.model_path,
                                   agent=agent if (frozen_agent is None or idx < 2) else frozen_agent)
                            for idx in range(players)]
        if type(players) == list:
            if len(players) < 2:
                raise ValueError("The game must have at least 2 players!")
            self.players = [Player(player["name"],
                                   model_path=self.model_path,
                                   agent=agent if (frozen_agent is None or idx < 2) else frozen_agent,
                                   mode=player["mode"],
                                   action_selection_mode=player.get("action_selection_mode",
                                                                    "random_dist")
                                   ) for idx, player in enumerate(players)]

        random.shuffle(self.players)
        self.monster_deck = Deck([Card(*monster) for monster in INITIAL_MONSTER_DECK])
        self.monster_deck.shuffle()
        self.dungeon_deck = Deck()
        self.adventurer = Adventurer()
        self.not_passed: list[Player] = self.players.copy()
        self.auto_run = auto_run
        self.game_log = []
        self.log_game = log_game
        self.verbose = verbose
        self.game_id = uuid.uuid4()
        self.rounds = 0
        self.phase_rounds = 0

        # utils for vorpal
        self.all_monsters_cards = list(set(self.monster_deck.cards))
        self.all_monsters_string = "\n".join(
            [
                f"{idx} - {monster}"
                for idx, monster in enumerate(self.all_monsters_cards)
            ]
        )
        self.all_monsters_options = [str(idx) for idx in range(len(self.monster_deck))]

    def take_turn(self, player: Player):
        """ This method will take a turn for the player. """
        turn_results = TURN_RESULTS.copy()

        if len(self.not_passed) == 1:
            return self._enter_the_dungeon(player, turn_results)

        else:
            if player.mode == "human" and not self.auto_run:
                action = ask_for_input(
                    "Will you [D]raw a card or [P]ass your turn?", ["D", "P"]
                )
            else:
                game_state = self.build_game_state(player)
                action_space = self.build_action_space(game_state)
                action = player.make_move(game_state, action_space)
                action_space[action] = 1
                if self.log_game:
                    self.game_log.append((game_state, action_space, {"reward": 0}))
                action = "D" if action == "draw" else "P"

            if action.upper() == "D":
                drawn_card = self.monster_deck.draw()
                if player.mode == "human" and not self.auto_run:
                    if self.adventurer.equipment:
                        action = ask_for_input(
                            f"Drawn card: {str(drawn_card)}.\n[A]dd to dungeon or [R]emove an equipment?",
                            ["A", "R"])
                    else:
                        action = "A"
                        self._print(f"There's no equipment left to remove! You've added {str(drawn_card)} to the dungeon!")
                else:
                    self._print(f"Player {player} drew {drawn_card}.")
                    game_state = self.build_game_state(player, card_drawn=drawn_card)
                    action_space = self.build_action_space(game_state)
                    
                    if self.adventurer.equipment:
                        action = player.make_move(game_state, action_space)
                        self._print(f"Player {player} chose to {action}.")
                    else:
                        action = "add_to_dungeon"
                        self._print(f"There's no equipment left to remove! Player {player} added {str(drawn_card)} to the dungeon!")
                    
                    action_space[action] = 1
                    if action != "add_to_dungeon":
                        equipment_taken = action
                        action = "R"
                    else:
                        self._print(f"{player} added {drawn_card} to the dungeon!")
                        equipment_taken = None
                        action = "A"

                if self.log_game:
                    self.game_log.append((game_state, action_space, {"reward": 0}))

                if action.upper() == "A":
                    player.cards_added.append(drawn_card)
                    self.dungeon_deck.add_card(drawn_card)
                else:
                    player.cards_taken.append(drawn_card)
                    equipment_list = "\n".join(self.adventurer.equipment)
                    equipment_options = [
                        equipment[1] for equipment in self.adventurer.equipment
                    ]

                    if player.mode == "human" and not self.auto_run:
                        action = ask_for_input(
                            f"Which equipment?\n{equipment_list}", equipment_options
                        )
                        removed_equipment = self.adventurer.equipment[
                            equipment_options.index(action.upper())]
                    else:
                        try:
                            equipment_taken = equipment_taken.replace("remove_", "")
                        except Exception as e:
                            print(action_space)
                            print(equipment_taken)
                            print(action)
                            print(player.equipment_taken)
                            print(self.adventurer.equipment)
                            print(game_state)
                            raise e

                        equipment_taken = "[" + equipment_taken[0].upper() + "]" + equipment_taken[1:]
                        # fix for vorpal
                        equipment_taken = "[V]orpal Sword" if equipment_taken == "[V]orpal" else equipment_taken
                        self._print(f"Player {player} took {equipment_taken}!")
                        removed_equipment = list(filter(lambda x: x == equipment_taken,
                                                        self.adventurer.equipment))[0]

                    player.equipment_taken.append(removed_equipment)
                    turn_results['equipment_taken'] = removed_equipment
                    self.adventurer.remove_equipment(removed_equipment)

                # the player that drew the last card must face the dungeon
                if not self.monster_deck:
                    return self._enter_the_dungeon(player, turn_results)
            else:
                self._print(f"Player {player} passed their turn!")
                turn_results['passed'] = True
                self.not_passed.remove(player)
            return turn_results

    def _enter_the_dungeon(self, player, turn_results):
        """ This method will handle the player entering the dungeon. """
        self._print(f"Player {player} is entering the dungeon!")
        if not self.auto_run:
            _ = input("Press Enter to continue...")
        self.dungeon_deck.shuffle()

        game_state = self.build_game_state(player)
        action_space = self.build_action_space(game_state, is_entering_dungeon=True)

        if Equipment.VORPAL_SWORD in self.adventurer.equipment:
            if player.mode == "human" and not self.auto_run:
                action = ask_for_input(
                    f"You have a Vorpal Sword. What will be its target?\n{self.all_monsters_string}",
                    self.all_monsters_options,
                )
            else:
                action = player.make_move(game_state, action_space).lower()
                action_space[action] = 1
                action = action.replace("target_", "")
                action = [monster.name.lower() for monster in self.all_monsters_cards].index(action)

            self.adventurer.vorpal_target = self.all_monsters_cards[int(action)]
            self._print("Vorpal target:", self.adventurer.vorpal_target)

        player_died = False
        while self.dungeon_deck:
            monster = self.dungeon_deck.draw()
            self._print(f"Facing monster: {monster}")

            monster_defeated = False
            if self.adventurer.vorpal_target and (
                self.adventurer.vorpal_target == monster
            ):
                self._print("You defeated the monster!")
                monster_defeated = True
            else:
                for equip in monster.dies_to:
                    if equip in self.adventurer.equipment:
                        self._print("You defeated the monster!")
                        monster_defeated = True
                        break

            if not monster_defeated:
                self._print(f"{monster.name} deals {monster.value} damage to you!")
                self.adventurer.take_damage(monster.value)
                if self.adventurer.hp <= 0:
                    
                    if self.log_game:
                        if Equipment.VORPAL_SWORD in self.adventurer.equipment:
                            # log vorpal targeting decision first so _reward_last_action patches it
                            self.game_log.append((game_state, action_space, {"reward": 0}))

                        # patch the most recent entry (vorpal if present, else pass/add)
                        self._reward_last_action(player, REWARDS["died"])
                            
                        # reward every other player for the player that died
                        for opponent in self.players:
                            if opponent != player and opponent.is_alive:
                                self._reward_last_action(opponent, REWARDS["passed"])

                    self._print("You died!")
                    turn_results['dungeon_result'] = 'died'
                    player_died = True
                    player.defeats += 1
                    if player.defeats > 1:
                        player.is_alive = False
                    break
            self._print(f"HP: {self.adventurer.hp}")
            if not self.auto_run:
                _ = input("Press Enter to continue...")

        if not player_died:

            if self.log_game:
                if Equipment.VORPAL_SWORD in self.adventurer.equipment:
                    # log vorpal targeting decision first so _reward_last_action patches it
                    self.game_log.append((game_state, action_space, {"reward": 0}))
                # patch the most recent entry (vorpal if present, else pass/add)
                self._reward_last_action(player, REWARDS["passed"])

                # # for each other player, punish them for the player that successfully passed the dungeon
                for opponent in self.players:
                    if opponent != player and opponent.is_alive:
                        self._reward_last_action(opponent, REWARDS["died"])

            turn_results['dungeon_result'] = 'passed'
            player.points += 1
            if player.points > 1:
                player.won = True
        
        if not self.auto_run:
            _ = input("Press Enter to continue...")
        
        return turn_results
    
    def _reward_last_action(self, player, reward):
        """ This method will reward the player for their last action. """
        for log in self.game_log[::-1]:
            if log[0]["current_player"] == player.name:
                log[2]["reward"] = reward
                break
    
    def _print(self, *args, **kwargs):
        """ Utility method to print the game state and actions. """
        if self.verbose:
            print(*args, **kwargs)
    
    def _reset(self):
        """ This method will reset the game state after a player enters the dungeon. """
        self.monster_deck = Deck([Card(*monster) for monster in INITIAL_MONSTER_DECK])
        self.monster_deck.shuffle()
        self.dungeon_deck = Deck()
        self.not_passed = self.players.copy()
        for player in self.players:
            player.cards_taken = []
            player.cards_added = []
            player.equipment_taken = []
            if not player.is_alive:
                self.not_passed.remove(player)
        self.adventurer = Adventurer()
    
    def _get_cards_taken(self, player):
        """ This method will return the cards taken by the player as a dictionary. """
        CARDS_MAPPING = {
            "Goblin": 0,
            "Skeleton": 0,
            "Orc": 0,
            "Vampire": 0,
            "Golem": 0,
            "Wraith": 0,
            "Demon": 0,
            "Dragon": 0
        }
        for card in player.cards_taken:
            CARDS_MAPPING[card.name] += 1
        return CARDS_MAPPING
    
    def _get_cards_added(self, player):
        """ This method will return the cards added by the player as a dictionary. """
        CARDS_MAPPING = {
            "Goblin": 0,
            "Skeleton": 0,
            "Orc": 0,
            "Vampire": 0,
            "Golem": 0,
            "Wraith": 0,
            "Demon": 0,
            "Dragon": 0
        }
        for card in player.cards_added:
            CARDS_MAPPING[card.name] += 1
        return CARDS_MAPPING
    
    def _get_equipment_taken(self, player):
        """ This method will return the equipment taken by the player as a dictionary. """
        EQUIPMENT_MAPPING = {
            "Armor": 0,
            "Shield": 0,
            "Chalice": 0,
            "Torch": 0,
            "Vorpal Sword": 0,
            "Lance": 0
        }
        for equipment in player.equipment_taken:
            EQUIPMENT_MAPPING[str(equipment)] += 1
        return EQUIPMENT_MAPPING

    def build_game_state(self, player: Player, card_drawn=None):
        """
        This method will build the game state for the player and return it as a
        dictionary. All information about the game is stored in a one-hot encoding
        format.

        Parameters
        ----------
        player : Player
            The player for which the game state is being built
        card_drawn : Card
            The card drawn by the player
        """
        game_state = {
            "game_id": self.game_id,
            "current_player": player.name,
            "rounds": self.rounds,
            "phase_rounds": self.phase_rounds,
            "defeats": player.defeats,
            "points": player.points,
            "deck_size": len(self.monster_deck),
            "dungeon_size": len(self.dungeon_deck),
            "adventurer_hp": self.adventurer.hp,
            "has_armor": Equipment.ARMOR in self.adventurer.equipment,
            "has_shield": Equipment.SHIELD in self.adventurer.equipment,
            "has_chalice": Equipment.CHALICE in self.adventurer.equipment,
            "has_torch": Equipment.TORCH in self.adventurer.equipment,
            "has_vorpal": Equipment.VORPAL_SWORD in self.adventurer.equipment,
            "has_lance": Equipment.LANCE in self.adventurer.equipment,
            "drew_goblin": 0,
            "drew_skeleton": 0,
            "drew_orc": 0,
            "drew_vampire": 0,
            "drew_golem": 0,
            "drew_wraith": 0,
            "drew_demon": 0,
            "drew_dragon": 0,
            "card_drawn": card_drawn
        }

        # card drawn
        if card_drawn:
            game_state["card_drawn"] = card_drawn.name
            game_state[f"drew_{card_drawn.name.lower()}"] = 1

        # player cards taken
        player_cards_taken = self._get_cards_taken(player)
        for card, count in player_cards_taken.items():
            game_state[f"{card}_taken"] = count

        # player cards added
        player_cards_added = self._get_cards_added(player)
        for card, count in player_cards_added.items():
            game_state[f"{card}_added"] = count
        
        # damage the adventurer can take based on the cards added by the player
        potential_damage = 0
        for card in player.cards_added:
            for equipment in card.dies_to:
                if equipment in self.adventurer.equipment:
                    potential_damage += card.value
        game_state["potential_damage"] = potential_damage

        # player equipment taken
        player_equipment_taken = self._get_equipment_taken(player)
        for equipment, count in player_equipment_taken.items():
            game_state[f"{equipment}_taken"] = count

        # get opponents states
        opponent_count = 0
        for opponent in self.players:
            if opponent != player:
                opponent_count += 1
                game_state[f"Opponent_{opponent_count}_defeats"] = opponent.defeats
                game_state[f"Opponent_{opponent_count}_points"] = opponent.points
                game_state[f"Opponent_{opponent_count}_has_passed"] = opponent not in self.not_passed
                game_state[f"Opponent_{opponent_count}_is_alive"] = opponent.is_alive
                opponent_equipment_taken = self._get_equipment_taken(opponent)
                for equipment, count in opponent_equipment_taken.items():
                    game_state[f"Opponent_{opponent_count}_{equipment}"] = count
        
        return game_state
    
    def build_action_space(self, game_state, is_entering_dungeon=False):
        """
        This method will build the action space for the player and return
        it as a dictionary. The action space contains all the possible actions
        the player can take in the game the moment it is called. Actions marked
        as -1 are not available to the player. All actions are in a one-hot
        encoding format.

        Parameters
        ----------
        game_state : dict
            The game state for the player
        is_entering_dungeon : bool
            If True, the player is entering the dungeon and the action space
            will be different
        """

        action_space = {
            "draw": -1,
            "pass": -1,
            "remove_armor": -1,
            "remove_shield": -1,
            "remove_chalice": -1,
            "remove_torch": -1,
            "remove_vorpal": -1,
            "remove_lance": -1,
            "add_to_dungeon": -1,
            "target_goblin": -1,
            "target_skeleton": -1,
            "target_orc": -1,
            "target_vampire": -1,
            "target_golem": -1,
            "target_wraith": -1,
            "target_demon": -1,
            "target_dragon": -1
        }

        # if the player is entering the dungeon and has the vorpal sword
        # the target actions are the only available actions
        if is_entering_dungeon and game_state["has_vorpal"]:
            action_space["target_goblin"] = 0
            action_space["target_skeleton"] = 0
            action_space["target_orc"] = 0
            action_space["target_vampire"] = 0
            action_space["target_golem"] = 0
            action_space["target_wraith"] = 0
            action_space["target_demon"] = 0
            action_space["target_dragon"] = 0
            return action_space

        # if the player did not draw a card yet, the draw and pass actions
        # are available
        if game_state["card_drawn"] is None:
            if game_state["deck_size"] > 0:
                action_space["draw"] = 0
                action_space["pass"] = 0
        else:
            # if the player drew a card, the add to dungeon and remove equipment
            # actions are available
            action_space["add_to_dungeon"] = 0
            action_space["remove_armor"] = 0 if game_state["has_armor"] else -1
            action_space["remove_shield"] = 0 if game_state["has_shield"] else -1
            action_space["remove_chalice"] = 0 if game_state["has_chalice"] else -1
            action_space["remove_torch"] = 0 if game_state["has_torch"] else -1
            action_space["remove_vorpal"] = 0 if game_state["has_vorpal"] else -1
            action_space["remove_lance"] = 0 if game_state["has_lance"] else -1
        
        return action_space

    def __str__(self):
        """ This method will return the game state as a string. """
        return f"""
        Players: {self.players}
        Monster Deck: {self.monster_deck}
        Dungeon Deck: {self.dungeon_deck}
        Adventurer: {self.adventurer}
        Round: {self.rounds}
        Not Passed: {self.not_passed}"""
    
    def run(self):
        """ This method will run the game loop. """

        starting_player = self.players[-1]
        # game loop
        while True:
            # get the next player in the queue
            self.players = list(roll(self.players, 1))
            current_player = self.players[0]

            if starting_player == current_player:
                self.rounds += 1
                self.phase_rounds += 1

            # if there's only one player alive, they win
            players_left = [p for p in self.players if p.is_alive]
            if len(players_left) == 1:
                self._print("Last player standing!")
                self._print(f"Player {players_left[0]} won!")
                if self.log_game:
                    # update the latest game log from this player to reflect the win
                    self._reward_last_action(players_left[0], REWARDS["won"])
                    # there's no need to update the rewards for the other players
                    # since they are not alive
                break
            if current_player in self.not_passed and current_player.is_alive:
                self._print(f"Player {current_player} is taking their turn!")
                turn_results = self.take_turn(current_player)
                if turn_results['dungeon_result'] is not None:
                    # Reset the phase rounds
                    self.phase_rounds = 0
                    
                    # The player that last faced the dungeon is the first in the
                    # queue for the next round. Compensate for the roll in the
                    # beginning of the loop
                    self.players = list(roll(self.players, -1))

                    self._print(f"Player {current_player} faced the dungeon and {turn_results['dungeon_result']}")

                    # print current scores and defeats
                    self._print("\nCurrent scores:\n----------------")
                    for player in self.players:
                        self._print(f"{player}: \t{player.points} points\t{player.defeats} defeats")
                    self._print("----------------\n")

                    if current_player.won:
                        self._print(f"Player {current_player} won by points!")
                        if self.log_game:
                            # give additional rewards to the player that won
                            self._reward_last_action(current_player, REWARDS["won"])
                            # for each other player, update the reward of their last action
                            for opponent in self.players:
                                if opponent != current_player and opponent.is_alive:
                                    self._reward_last_action(opponent, REWARDS["lost"])
                        break
                    else:
                        if not current_player.is_alive:
                            # update the player's last action to reflect the loss
                            self._print(f"Player {current_player} is out of the game!")
                            if self.log_game:
                                self._reward_last_action(current_player, REWARDS["lost"])
                        self._reset()
                    continue
        self._print("Game over!")


                        
