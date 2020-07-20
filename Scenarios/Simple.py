"""
Written by Benjamin Raiford
Modeled after multiagent-particle-envs (MPE) from OpenAI
7/19/2020
"""

import numpy as np

# check for commitment

class Agent:
    # Init method
    def __init__(self):
        self.location = -1
        self.number = -1

        sight_options = ['self', 'others', 'all']
        self.sight = sight_options[1]


class World:
    # Init method
    def __init__(self):
        self.agents = []
        self.line = []


class Scenario:
    def make_world(self, n_agents=3):
        world = World()

        world.agents = [Agent() for i in range(n_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'Agent {}'.format(i)
            agent.sight = 'others'

        self.reset_world(world, n_agents)
        return world

    def reset_world(self, world, n_agents):
        world.line = [None] * n_agents

        initial_ordering = np.random.choice(n_agents, size=n_agents, replace=False)
        print("DEBUG: initial_ordering:", initial_ordering)

        for i, agent in enumerate(world.agents):
            agent.number = np.random.randint(1, 100, dtype=int)

            agent.location = initial_ordering[i]
            world.line[agent.location] = agent

        line_numbers = [a.number for a in world.line]
        print("DEBUG: line_numbers:", line_numbers)

    def check_done(self, world):
        done = True
        for i in range(len(world.line) - 1):
            if world.line[i].number > world.line[i+1].number:
                done = False
        return done

    def reward(self, agent, world):
        pass

# Entry point
if __name__ == "__main__":
    n = Scenario()
    nw = n.make_world()
    print(n.check_done(nw))
