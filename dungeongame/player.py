from dungeongame.card import Card
import random
import numpy as np

from dungeongame.model import Agent

GAMESTATE_COLUMN_ORDER = ['rounds', 'phase_rounds', 'defeats',
       'points', 'deck_size', 'dungeon_size', 'adventurer_hp', 'has_armor',
       'has_shield', 'has_chalice', 'has_torch', 'has_vorpal', 'has_lance',
       'drew_goblin', 'drew_skeleton', 'drew_orc', 'drew_vampire',
       'drew_golem', 'drew_wraith', 'drew_demon', 'drew_dragon',
       'Goblin_taken', 'Skeleton_taken', 'Orc_taken', 'Vampire_taken',
       'Golem_taken', 'Wraith_taken', 'Demon_taken', 'Dragon_taken',
       'Goblin_added', 'Skeleton_added', 'Orc_added', 'Vampire_added',
       'Golem_added', 'Wraith_added', 'Demon_added', 'Dragon_added',
       'potential_damage', 'Armor_taken', 'Shield_taken', 'Chalice_taken',
       'Torch_taken', 'Vorpal Sword_taken', 'Lance_taken',
       'Opponent_1_defeats', 'Opponent_1_points', 'Opponent_1_has_passed', 'Opponent_1_is_alive',
       'Opponent_1_Armor', 'Opponent_1_Shield', 'Opponent_1_Chalice',
       'Opponent_1_Torch', 'Opponent_1_Vorpal Sword', 'Opponent_1_Lance',
       'Opponent_2_defeats', 'Opponent_2_points', 'Opponent_2_has_passed', 'Opponent_2_is_alive',
       'Opponent_2_Armor', 'Opponent_2_Shield', 'Opponent_2_Chalice',
       'Opponent_2_Torch', 'Opponent_2_Vorpal Sword', 'Opponent_2_Lance',
       'Opponent_3_defeats', 'Opponent_3_points', 'Opponent_3_has_passed', 'Opponent_3_is_alive',
       'Opponent_3_Armor', 'Opponent_3_Shield', 'Opponent_3_Chalice',
       'Opponent_3_Torch', 'Opponent_3_Vorpal Sword', 'Opponent_3_Lance']

ACTION_SPACE_COLUMN_ORDER =  ['draw', 'pass',
       'remove_armor', 'remove_shield', 'remove_chalice', 'remove_torch',
       'remove_vorpal', 'remove_lance', 'add_to_dungeon', 'target_goblin',
       'target_skeleton', 'target_orc', 'target_vampire', 'target_golem',
       'target_wraith', 'target_demon', 'target_dragon']

MAX_FEATURE_VALUES = np.array([37.,  6.,  1.,  1., 13., 34., 11.,  1.,  1.,  1.,  1.,  1.,  1.,
        1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  2.,  2.,  1.,  1.,  2.,
        1.,  1.,  1.,  2.,  2.,  2.,  2.,  2.,  1.,  1.,  1., 16.,  1.,
        1.,  1.,  1.,  1.,  1.,  2.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,
        1.,  1.,  2.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  2.,
        1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.], dtype=np.float32)

MIN_FEATURE_VALUES = np.array([1., 0., 0., 0., 0., 0., 3., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0.], dtype=np.float32)


class Player:

    def __init__(self, name: str, mode="random", model_path=None, agent=None,
                 action_selection_mode="random_dist"):
        self.points: int = 0
        self.cards_taken: list[Card] = []
        self.cards_added: list[Card] = []
        self.equipment_taken: list[Card] = []
        self.defeats: int = 0
        self.is_alive: bool = True
        self.won: bool = False
        self.name: str = name
        self.model = agent
        self.mode = mode
        self.min_feature_values = MIN_FEATURE_VALUES
        self.max_feature_values = MAX_FEATURE_VALUES
        self.model_path = model_path
        self.action_selection_mode = action_selection_mode
    
    def __str__(self):
        return self.name
    
    def __repr__(self):
        return f"Name: {self.name}, Points: {self.points}, Defeats: {self.defeats}, Cards taken: {[str(card) for card in self.cards_taken]}"
    
    def make_random_move(self, action_space):
        """ Make a random move based on the current action space. """
        # select a random valid action
        valid_actions = [action for action in action_space if action_space[action] == 0]

        # fix the action weight for the draw and pass action
        if action_space['draw'] == 0 and action_space['pass'] == 0:
            valid_actions = valid_actions + ['draw'] * 2

        # fix the action weight for the add_to_dungeon action
        elif action_space['add_to_dungeon'] == 0:
            # add the add_to_dungeon action multiple times to increase its weight
            valid_actions = valid_actions + ['add_to_dungeon'] * (len(valid_actions) - 1)

        if valid_actions:
            return random.choice(valid_actions)
    
    def _load_model(self):
        self.model = Agent(input_shape=(len(GAMESTATE_COLUMN_ORDER),), num_actions=len(ACTION_SPACE_COLUMN_ORDER))
        if self.model_path is not None:
            self.model.load(self.model_path)

    def _minmax_scaling(self, x):
        return (x - MIN_FEATURE_VALUES) / (MAX_FEATURE_VALUES - MIN_FEATURE_VALUES)
    
    def _transform_model_input(self, input):
        # transform each boolean value on the input into a 0 or 1
        new_input = []
        for value in input:
            if type(value) == bool:
                new_input.append(1 if value else 0)
            else:
                new_input.append(value)
        return self._minmax_scaling(np.array([new_input], dtype=np.float32))
        
    
    def make_greedy_move(self, game_state, action_space):
        """
        Uses the model to predict the reward for each possible action and
        selects the action that maximizes the reward.
        """

        if self.model is None:
            self._load_model()

        # reorder the game state columns
        game_state = [game_state[column] for column in GAMESTATE_COLUMN_ORDER]

        # get the valid actions
        valid_actions = np.zeros(len(ACTION_SPACE_COLUMN_ORDER))
        for i, action in enumerate(ACTION_SPACE_COLUMN_ORDER):
            if action_space[action] == 0:
                valid_actions[i] = 1

        # get the expected reward for each action
        selected_action = self.model.select_action(
            self._transform_model_input(game_state),
            valid_actions,
            self.action_selection_mode)
        
        # select the action that maximizes the reward
        selected_action = ACTION_SPACE_COLUMN_ORDER[selected_action]

        # check if the selected action is valid
        if action_space[selected_action] == 0:
            return selected_action
        else:
            # if the selected action is not valid, raise an error
            raise ValueError("Invalid action selected.")
        
    
    def make_move(self, game_state, action_space):
        """
        Make a move based on the current game state and action space. The mode 
        parameter can be used to specify the policy for selecting an action.

        Parameters
        ----------
        game_state : dict
            A dictionary containing the current game state.
        action_space : dict
            A dictionary containing the current action space.
        """
        if self.mode == "random":
            return self.make_random_move(action_space)
        elif self.mode == "greedy":
            return self.make_greedy_move(game_state, action_space)
        else:
            raise ValueError("Invalid mode. The possible values are 'random' and 'greedy'.")

        

