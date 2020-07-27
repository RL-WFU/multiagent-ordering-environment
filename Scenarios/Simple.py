"""
Written by Benjamin Raiford
Modeled after multiagent-particle-envs (MPE) from OpenAI

Initial Action class and reward function written by Frank Liu

Updated 7/26/20 by Ben Raiford
"""

import numpy as np


class Agent:
    # Init method
    def __init__(self):
        self.column = -1
        self.row = -1
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
        # Communication (for potential later implementation)
        # NOT IMPLEMENTED
        self.communication = False


class World:
    # Init method
    def __init__(self, n_agents):
        # Set num_agents and declare agents array
        self.num_agents = n_agents
        self.agents = []

        # Create array environment for agents to order themselves in
        self.height = 3
        self.width = n_agents
        self.line = np.empty((self.height, self.width), Agent)

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
        return self

    # Resets the world (for use after every episode)
    def reset_world(self):
        # Reset the time step
        self.timestep = 0

        for i, agent in enumerate(self.agents):
            # Assign each agent a random number
            agent.number = np.random.randint(1, 100, dtype=int)
            # Set each agent's location to its initial position
            agent.column = i
            agent.row = 1

            # Update world with agents initial location
            self.line[agent.row, agent.column] = agent

        self.debug_print_line()

    def check_done(self):
        done = True
        filled = True

        for i in range(self.num_agents):
            if not self.line[1, i]:
                print("The line is not filled")
                filled = False
                done = False

            if not filled:
                break

        if filled:
            for i in range(self.num_agents - 1):
                if self.line[1, i].number > self.line[1, i + 1].number:
                    done = False

        return done

    def debug_print_line(self, show_row=1):
        line_numbers = [a.number for a in self.line[1,]]
        print("DEBUG: line_numbers:", line_numbers)

    # Agent movement mechanic
    def move(self, agent):

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
                if not self.line[original_row, original_column - 1]:
                    # Change agent's location and clear the original location
                    agent.column = agent.column - 1
                    self.line[original_row, original_column] = None

        # Right
        elif agent.action.direction == 2:
            # If moving agent is not in right-most column
            if agent.column is not self.num_agents - 1:
                # If space is not occupied
                if not self.line[original_row, original_column + 1]:
                    # Change agent's location and clear the original location
                    agent.column = agent.column + 1
                    self.line[original_row, original_column] = None

        # Up
        elif agent.action.direction == 3:
            # If moving agent is not in top row
            if agent.row is not 0:
                # If space is not occupied
                if not self.line[original_row - 1, original_column]:
                    # Change agent's location and clear the original location
                    agent.row = agent.row - 1
                    self.line[original_row, original_column] = None

        # Down
        elif agent.action.direction == 4:
            # If moving agent is not in bottom row
            if agent.row is not 2:
                # If space is not occupied
                if not self.line[original_row + 1, original_column]:
                    # Change agent's location and clear the original location
                    agent.row = agent.row + 1
                    self.line[original_row, original_column] = None

        # Update world with agent's new location
        self.line[agent.row, agent.column] = agent

    # Step function
    def step(self, agent_actions):
        # Assert that there is an action for every agent (and no more)
        assert (len(agent_actions) == self.num_agents)

        # Initialize reward
        reward = 0

        # For every agent
        for i, agent in enumerate(self.agents):
            # Get agent's action from the array
            agent.action.direction = agent_actions[i]
            # Move
            self.move(agent)
            # DEBUG: Print current array
            print(self.line, "\n")

        # Check if the environment is done
        done = self.check_done()

        # Time step and negative reward
        # FIXME: make sure this shouldn't happen on the last iteration as well
        if not done:
            self.timestep += 1
            reward = -1

        # Return reward and done
        return reward, done

# Entry point
if __name__ == "__main__":
    # set parameters
    num_agents = 2

    # initialize environment
    env = World(num_agents)
    env.make_world()

    # declare policy array
    policy = np.zeros(num_agents)
    # fill policy array with random
    for i in range(num_agents):
        policy[i] = np.random.randint(5)

    print(policy)
    print(env.step(policy))
