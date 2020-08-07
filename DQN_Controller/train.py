import itertools
from copy import copy
import numpy as np
import tensorflow as tf
from Scenarios.Simple import World as SimpleWorld
from DQN_Controller.Agent import DQNAgent
from DQN_Controller.ReplayBuffer import ReplayBuffer


# Process maps to input into a Tensorflow Model
def process_map(orig_map, map_height, map_width, simplify_agents=False):
    if simplify_agents:
        new_map = np.zeros_like(orig_map, dtype=float)
        for row in range(map_height):
            for column in range(map_width):
                if orig_map[row, column]:
                    new_map[row, column] = orig_map[row, column].number

        new_map = new_map.reshape((map_height, map_width, 1))
        # new_map = new_map.reshape(map_height*map_width, 1)
        return new_map

    else:
        orig_map = orig_map.reshape((map_height, map_width, 1))
        return orig_map


# Code to process actions. This will allow the DQN to use integers associated with an action
def process_actions(num_agents, num_actions):
    # Recursive function that builds arrays of actions with num_agents length, then appends them to an array
    def build_actions(actions, single_action=np.zeros(num_agents), index=0):

        # When an action as been built
        if index >= num_agents:
            # Append a copy of the built action to the action list
            actions.append(copy(single_action))  # MUST be a copy...
            return

        # For every action
        for i in range(num_actions):
            # Update number at current index
            single_action[index] = i
            # Increase index by one, make recursive call
            build_actions(actions, single_action, index + 1)

    # Initialize an empty list of actions
    action_list = []
    # Fill the list with every possible action
    build_actions(action_list)

    # Make numpy for utility, and reshape to make more readable
    action_list = np.array(action_list)
    # action_list.reshape(num_actions ** num_agents, num_agents, 1)

    return action_list


def run_environment():
    episodes = 2000

    env = SimpleWorld(2)
    env.make_world()

    integers_to_actions = process_actions(env.num_agents, 5)

    my_agent = DQNAgent((env.height, env.width), 5**env.num_agents)

    for episode in range(episodes):
        done = False
        # env.reset_world()
        state = process_map(env.reset_world(), env.height, env.width, simplify_agents=True)  # can switch to true here
        total_reward = 0

        while not done:
            # Process action
            action_index = my_agent.act(state)
            action = integers_to_actions[action_index]

            # Process next_state
            new_state, reward, done = env.step_global(action)
            next_state = process_map(new_state, env.height, env.width, simplify_agents=True)

            # Add experience and train
            my_agent.update_memory(state, action_index, reward, next_state, done)
            if my_agent.replay.count > my_agent.minimum_buffer:
                my_agent.train()


            state = next_state
            total_reward += reward

            if total_reward <= -200:
                break


        print("Total reward after episode {} is {} and epsilon is {}. Done = {}. Replay Buffer length: {}"
              .format(episode, total_reward, my_agent.epsilon, done, my_agent.replay.count))



# Entry point
if __name__ == "__main__":
    run_environment()
