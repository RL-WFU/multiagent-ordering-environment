"""
Written by Benjamin Raiford
Modeled after multiagent-particle-envs (MPE) from OpenAI
7/19/2020
"""

import numpy as np


# modified by Frank Liu for Action class and reward function

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
        action_space = ['left', 'right', 'up', 'down', 'stay']
        self.action = action_space[-1]

    def move(self, agent, world):
        current_column = agent.column
        current_row = agent.row

        # Left
        if agent.action == 'left':
            if not world.line[current_row, current_column - 1]:
                agent.column = agent.column - 1
        # Right
        elif agent.action == 'right':
            if not world.line[current_row, current_column + 1]:
                agent.column = agent.column + 1
        # Up
        elif agent.action == 'up':
            if not world.line[current_row - 1, current_column]:
                agent.row = agent.row - 1
        # Down
        elif agent.action == 'down':
            if not world.line[current_row + 1, current_column]:
                agent.row = agent.row + 1
        # Any code for stay would be redundant

    def act(self, world):
        # FIXME
        for agent in enumerate(world.agents):
            agent.action.move(agent, world)
            world.timestep += 1


class World:
    # Init method
    def __init__(self, n_agents):
        self.num_agents = n_agents
        self.agents = []
        self.line = np.empty((3, n_agents), Agent)
        self.timestep = 0

    # FIXME: next need to build step to make algorithms work
    def step(self):
        for agent in self.agents:
            agent.action = 'stay'

class Scenario:
    def make_world(self, n_agents=3):
        world = World(n_agents)
        world.agents = [Agent() for i in range(n_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'Agent {}'.format(i)
            agent.sight = 'others'

        self.reset_world(world, n_agents)
        return world

    def reset_world(self, world, n_agents):

        world.timestep = 0

        for i, agent in enumerate(world.agents):
            agent.number = np.random.randint(1, 100, dtype=int)
            agent.column = i
            agent.row = 1

            world.line[agent.row, agent.column] = agent

        line_numbers = [a.number for a in world.line[1, ]]
        print("DEBUG: line_numbers:", line_numbers)

    def check_done(self, world):
        done = True
        filled = True

        for i in range(world.num_agents - 1):
            if not world.line[1, i]:
                print("The line is not filled")
                filled = False
                done = False

            if not filled:
                break

        if filled:
            for i in range(world.num_agents - 1):
                if world.line[1, i].number > world.line[1, i + 1].number:
                    done = False

        return done

    def reward(self, agent, world):
        return world.timestep * -1


# Entry point
if __name__ == "__main__":
    scen = Scenario()
    env = scen.make_world(2)
    scen.check_done(env)
