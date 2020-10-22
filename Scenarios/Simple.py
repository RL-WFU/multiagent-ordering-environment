"""
Written by Benjamin Raiford
Modeled after multiagent-particle-envs (MPE) from OpenAI

Initial Action class and reward function written by Frank Liu

Updated 7/27/20 by Ben Raiford
"""

import numpy as np


class Agent:
    # Init method
    def __init__(self):
        # Agent location
        self.column = -1
        self.row = -1

        # Agent number
        self.number = -1

        sight_options = ['self', 'others', 'all']
        self.sight = sight_options[1]
        self.action = Action()


class Action:
    def __init__(self):
        """
        Possible directions
            0: Stay
            1: Left
            2: Right
            3: Up
            4: Down
        """
        self.direction = -1

        # TODO: Implement communication


class World:
    # Init method
    def __init__(self, n_agents):
        # Set num_agents and declare agents array
        self.num_agents = n_agents
        self.agents = []

        # Create array environment for agents to order themselves in
        self.height = 3
        self.width = n_agents
        self.map = None
        self.initialized_done = None

        # Global time counter
        self.timestep = -1

    # Makes world (for use at the beginning)
    def make_world(self):
        # Create agents to fill the agents array
        self.agents = [Agent() for i in range(self.num_agents)]
        for i, agent in enumerate(self.agents):
            agent.name = 'Agent {}'.format(i)
            agent.sight = 'others'

        self.reset_world()

    # Resets the world (for use after every episode)
    def reset_world(self):
        # Reset the time step
        self.timestep = 0

        agent_numbers = np.random.choice(np.arange(self.num_agents)+1, self.num_agents, replace=False)
        print("AGENT NUMBERS", agent_numbers)

        # Empties world
        self.map = np.empty((self.height, self.width), Agent)

        for i, agent in enumerate(self.agents):
            # Assign each agent a random number
            agent.number = agent_numbers[i]
            # agent.number = float(agent.number)
            # Set each agent's location to its initial position
            agent.column = i
            agent.row = 1

            # Adds agents to their initial location
            self.map[agent.row, agent.column] = agent

            self.initialized_done = self.check_done()

        return self.map

    def create_observation(self):
        obs_map = np.zeros_like(self.map)
        for row in range(self.height):
            for column in range(self.width):
                # If an agent is in the map location, print the number instead of E
                if self.map[row, column]:
                    obs_map[row, column] = self.map[row, column].number
                else:
                    self.map[row, column] = 0

        return obs_map


    # Check if the world is done
    def check_done(self):
        done = True
        filled = True

        for i in range(self.num_agents):
            if not self.map[1, i]:
                # print("DEBUG: The line is not filled")
                filled = False
                done = False

            if not filled:
                break

        if filled:
            for i in range(self.num_agents - 1):
                if self.map[1, i].number > self.map[1, i + 1].number:
                    done = False

        return done

    # Agent movement mechanic
    def move(self, agent):
        # Keep track of the agent's original position
        original_column = agent.column
        original_row = agent.row

        # Stay
        if agent.action.direction == 0:
            return

        # Left
        if agent.action.direction == 1:
            # If moving agent is not in the left-most column
            if agent.column is not 0:
                # If space is not occupied
                if not self.map[original_row, original_column - 1]:
                    # Change agent's location and clear the original location
                    agent.column = agent.column - 1
                    self.map[original_row, original_column] = None

        # Right
        elif agent.action.direction == 2:
            # If moving agent is not in right-most column
            if agent.column is not self.num_agents - 1:
                # If space is not occupied
                if not self.map[original_row, original_column + 1]:
                    # Change agent's location and clear the original location
                    agent.column = agent.column + 1
                    self.map[original_row, original_column] = None

        # Up
        elif agent.action.direction == 3:
            # If moving agent is not in top row
            if agent.row is not 0:
                # If space is not occupied
                if not self.map[original_row - 1, original_column]:
                    # Change agent's location and clear the original location
                    agent.row = agent.row - 1
                    self.map[original_row, original_column] = None

        # Down
        elif agent.action.direction == 4:
            # If moving agent is not in bottom row
            if agent.row is not 2:
                # If space is not occupied
                if not self.map[original_row + 1, original_column]:
                    # Change agent's location and clear the original location
                    agent.row = agent.row + 1
                    self.map[original_row, original_column] = None

        # Update world with agent's new location
        self.map[agent.row, agent.column] = agent

    # Step function
    def step_global(self, agent_actions):
        # Assert that there is an action for every agent (and no more)
        assert (len(agent_actions) == self.num_agents)

        # Initialize reward for done state
        reward = 0

        # For every agent
        for i, agent in enumerate(self.agents):
            # Get agent's action from the array
            agent.action.direction = agent_actions[i]
            # Move
            self.move(agent)

        # Check if the environment is done
        done = self.check_done()

        # Time step and negative reward
        if not done:
            self.timestep += 1
            reward = -1

        # Return reward and done
        return self.map, reward, done

    def render(self):
        # Create a map to print (that won't raise issues with NoneType), initialize every spot with E
        printed_map = np.full_like(self.map, 'E')

        # Iterate through the map
        for row in range(self.height):
            for column in range(self.width):
                # If an agent is in the map location, print the number instead of E
                if self.map[row, column]:
                    printed_map[row, column] = self.map[row, column].number

        # Output the map
        print(printed_map, "\n")

def random_policy(length):
    # declare policy array
    policy = np.zeros(length)
    # fill policy array with random
    for i in range(length):
        policy[i] = np.random.randint(5)

    return policy


def test_harness(n_agents=3):
    # initialize environment
    env = World(n_agents)
    env.make_world()

    # Original
    env.render()

    # Step 1
    policy1 = random_policy(n_agents)
    print("Policy 1:", policy1)
    env.step_global(policy1)
    env.render()

    # Step 2
    policy2 = random_policy(n_agents)
    print("Policy 2:", policy2)
    env.step_global(policy2)
    env.render()

    # Step3
    policy3 = random_policy(n_agents)
    print("Policy 3:", policy3)
    env.step_global(policy3)
    env.render()


# Entry point
if __name__ == "__main__":
    # test_harness(3)
    env = World(3)
    env.make_world()
    print(env.reset_world())
    print(env.create_observation())
