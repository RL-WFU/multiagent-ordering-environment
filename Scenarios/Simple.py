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
        self.location = -1
        self.number = -1

        sight_options = ['self', 'others', 'all']
        self.sight = sight_options[1]
        self.action = Action()

class Action:
    def __init__(self):
        action_space = ['left','right','stay']
        self.action = action_space[2]

    def move(self, agent, world):
        if agent.action == 'left':
            tmp_agent = world.line[agent.location]
            world.line[agent.location] = world.line[agent.location - 1]
            world.line[agent.location - 1] = tmp_agent
        elif agent.action == 'right':
            tmp_agent = world.line[agent.location]
            world.line[agent.location] = world.line[agent.location + 1]
            world.line[agent.location + 1] = tmp_agent

    def act(self, world):
        for agent in enumerate(world.agents):
            agent.action.move(agent,world)
            world.timestep += 1


class World:
    # Init method
    def __init__(self):
        self.agents = []
        self.line = []
        self.timestep = 0



class Scenario:
    def make_world(self, n_agents=3):
        world = World()
        world.timestep = 0
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
        return world.timestep * -1

# Entry point
if __name__ == "__main__":
    n = Scenario()
    nw = n.make_world()
    print(n.check_done(nw))
