import keras
import tensorflow as tf
import numpy as np

# To create the model for our agent, we need two neural networks: the policy
# network and the value network. The policy network will be used to select the
# best action to take, while the value network will be used to estimate the
# value of a given state.

# The policy network will take the current game state as input and output a
# probability distribution over the possible actions. The value network will
# take the current game state as input and output a single value representing
# the expected reward for that state.

ACTION_SPACE_COLUMN_ORDER =  ['draw', 'pass',
       'remove_armor', 'remove_shield', 'remove_chalice', 'remove_torch',
       'remove_vorpal', 'remove_lance', 'add_to_dungeon', 'target_goblin',
       'target_skeleton', 'target_orc', 'target_vampire', 'target_golem',
       'target_wraith', 'target_demon', 'target_dragon']

# class for the policy network
class PolicyNetwork(keras.Model):
    def __init__(self, input_shape, num_actions):
        super(PolicyNetwork, self).__init__()
        self.dense1 = keras.layers.Dense(128, activation='relu', input_shape=input_shape)
        self.dense2 = keras.layers.Dense(64, activation='relu')
        self.dense3 = keras.layers.Dense(num_actions, activation='linear')

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return self.dense3(x)


# class for the value network
class ValueNetwork(keras.Model):
    def __init__(self, input_shape):
        super(ValueNetwork, self).__init__()
        self.dense1 = keras.layers.Dense(128, activation='relu', input_shape=input_shape)
        self.dense2 = keras.layers.Dense(64, activation='relu')
        self.dense3 = keras.layers.Dense(1, activation='linear')

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return self.dense3(x)
    
    
# class for the agent
class Agent:
    def __init__(self, input_shape, num_actions, gamma=0.9):
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.policy_network = PolicyNetwork(input_shape, num_actions)
        self.value_network = ValueNetwork(input_shape)
        self.policy_optimizer = keras.optimizers.Adam(learning_rate=3e-4)
        self.value_optimizer = keras.optimizers.Adam(learning_rate=3e-4)
        self.gamma = gamma
        
    def select_action(self, state, valid_actions, mode="random_dist"):
        """
        Select an action based on the current state and valid actions.

        Parameters
        ----------
        state : np.array
            The current state of the game. This should be a 1D array of shape
            (input_shape[0],).
        valid_actions : np.array
            A 1D array of shape (num_actions,) where 1 indicates a valid action
            and 0 indicates an invalid action.
        mode : str
            The mode to use for selecting the action. The possible values are
            'random_dist' and 'max_probability'.
        """
        # get the action probabilities from the policy network
        action_probs = self.policy_network(state)

        # apply softmax to the valid action probabilities
        action_probs = tf.nn.softmax(action_probs) + tf.keras.backend.epsilon()

        # mask out invalid actions
        valid_action_probs = action_probs * valid_actions

        # Normalize probabilities after masking
        valid_action_probs = valid_action_probs / tf.reduce_sum(valid_action_probs)

        # # print the action probabilities for each action in ACTION_SPACE_COLUMN_ORDER
        # for i, action in enumerate(ACTION_SPACE_COLUMN_ORDER):
        #     print(f"{action}: {valid_action_probs[0][i]}")

        # add a small value to prevent log(0)
        valid_action_probs = valid_action_probs + tf.keras.backend.epsilon()

        # if there's nan values, raise an error
        if tf.reduce_any(tf.math.is_nan(valid_action_probs)):
            print("Action probabilities", action_probs)
            print("Valid actions", valid_actions)
            print("Valid action probs", valid_action_probs)
            raise ValueError("Action probabilities contain NaN values")

        # ----- RANDOM ACTION -----
        # sample an action from the action probabilities
        if mode == "random_dist":
            action = tf.random.categorical(tf.math.log(valid_action_probs), 1)

            # if action is invalid, choose again
            while valid_actions[action.numpy()[0][0]] == 0:
                print("Action probs", valid_action_probs)
                action = tf.random.categorical(tf.math.log(valid_action_probs), 1)
            return action.numpy()[0][0]
        # ----- RANDOM ACTION -----

        # ----- MAX PROBABILITY ACTION -----
        # # select the action with the highest probability
        elif mode == "max_probability":
            action = tf.argmax(valid_action_probs, axis=1)
            return action.numpy()[0]
        # ----- MAX PROBABILITY ACTION -----

        # if the mode is not recognized, raise an error
        else:
            raise ValueError("Invalid mode. The possible values are 'random' and 'max_probability'.")
        

    def train(self, states, actions, discounted_rewards):
        """
        Train the agent on a batch of experiences.

        Parameters
        ----------
        states : np.array
            A 2D array of shape (batch_size, input_shape[0]) containing the
            states.
        actions : np.array
            A 2D array of shape (batch_size, num_actions) containing the
            actions taken.
        discounted_rewards : np.array
            A 1D array of shape (batch_size,) containing the discounted
            returns G_t for each timestep.
        """
        # convert states to tensor
        states = tf.convert_to_tensor(states, dtype=tf.float32)

        # convert actions to tensor
        actions = tf.convert_to_tensor(actions, dtype=tf.int32)

        # convert discounted rewards to tensor
        discounted_rewards = tf.convert_to_tensor(discounted_rewards, dtype=tf.float32)

        # normalize discounted rewards
        eps = tf.keras.backend.epsilon()
        discounted_rewards = (discounted_rewards - tf.reduce_mean(discounted_rewards)) / (tf.math.reduce_std(discounted_rewards) + eps)

        # train the policy network
        policy_loss = self._train_policy_network(states, actions, discounted_rewards)
        # train the value network
        value_loss = self._train_value_network(states, discounted_rewards)

        # return the loss for plotting
        return policy_loss, value_loss

    # def _get_discounted_rewards(self, rewards):
    #     discounted_rewards = np.zeros_like(rewards, dtype=np.float32)
    #     running_add = 0
    #     for t in reversed(range(len(rewards))):
    #         running_add = running_add * self.gamma + rewards[t]
    #         discounted_rewards[t] = running_add
    #     return discounted_rewards

    def _train_policy_network(self, states, actions, discounted_rewards):
        with tf.GradientTape() as tape:
            # get the action probabilities from the policy network
            action_probs = self.policy_network(states)

            # apply softmax to the action probabilities
            action_probs = tf.nn.softmax(action_probs)

            # mask out action that were not taken
            actions = np.array(actions, dtype=np.float32)
            selected_action_probs = action_probs * actions

            # sum over the actions to get the probabilities of the selected actions
            selected_action_probs = tf.reduce_sum(selected_action_probs, axis=1)

            # add a small value to prevent log(0)
            eps = tf.keras.backend.epsilon()
            selected_action_probs = selected_action_probs + eps
            log_probs = tf.math.log(selected_action_probs)

            # get the estimated values from the value network
            values = self.value_network(states)

            # Monte Carlo advantage: how much better was this action than expected?
            advantages = discounted_rewards - values

            # calculate the policy loss
            policy_loss = -tf.reduce_mean(log_probs * advantages)

        # calculate the gradients of the policy loss
        policy_grads = tape.gradient(policy_loss, self.policy_network.trainable_variables)

        # apply the gradients to the policy network
        self.policy_optimizer.apply_gradients(zip(policy_grads, self.policy_network.trainable_variables))
        return policy_loss

    def _train_value_network(self, states, discounted_rewards):
        with tf.GradientTape() as tape:
            # get the predicted values from the value network
            values = self.value_network(states)
            # calculate the value loss
            value_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)(values, discounted_rewards)
        # calculate the gradients of the value loss
        value_grads = tape.gradient(value_loss, self.value_network.trainable_variables)
        # apply the gradients to the value network
        self.value_optimizer.apply_gradients(zip(value_grads, self.value_network.trainable_variables))
        return value_loss
    
    def save(self, path):
        self.policy_network.save_weights(path + '/policy_network.weights.h5')
        self.value_network.save_weights(path + '/value_network.weights.h5')
    
    def load(self, path):
        self.policy_network.build((None, self.input_shape[0]))
        self.value_network.build((None, self.input_shape[0]))
        self.policy_network.load_weights(path + '/policy_network.weights.h5')
        self.value_network.load_weights(path + '/value_network.weights.h5')


if __name__ == "__main__":
    # create some dummy data for training
    example_states = np.array([[1, 2, 3, 4, 4, 3, 2, 1],
                       [5, 6, 7, 8, 5, 6, 7, 8],
                       [9, 10, 11, 12, 12, 12, 12, 12]], dtype=np.float32)

    example_actions = np.array([[0, 0, 1],
                        [0, 1, 0],
                        [1, 0, 0]], dtype=np.int32)

    example_rewards = np.array([1, 2, 3], dtype=np.float32)

    # create an instance of the agent
    agent = Agent(input_shape=(example_states.shape[1],), num_actions=example_actions.shape[1])

    # train the agent
    agent.train(example_states, example_actions, example_rewards)

    # select an action using the agent
    example_state = tf.convert_to_tensor([[5, 6, 7, 8, 5, 6, 7, 8]], dtype=tf.float32)
    example_action = agent.select_action(example_state, np.array([[1, 0, 0]])).numpy()
    print(example_action)