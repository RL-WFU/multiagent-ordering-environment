from DQN_Controller.train import process_map
from Scenarios.Simple import World as SimpleWorld
from DQN_Controller.Agent import DQNAgent


def run_individuals():
    episodes = 2000
    individual_agents = 2

    env = SimpleWorld(individual_agents)
    env.make_world()

    all_networks = [DQNAgent((env.height, env.width), 5) for i in range(individual_agents)]

    # Statistics to graph later
    rewards_for_save = []
    epsilons_for_save = []
    dones_for_save = []

    for episode in range(episodes):
        done = False
        state = process_map(env.reset_world(), env.height, env.width, simplify_agents=True)  # can switch to true here
        total_reward = 0

        while not done:
            # Process action
            action = []
            # Append each agent's action to the state
            for agent in all_networks:
                action.append(agent.act(state))

            # Process next_state
            new_state, reward, done = env.step_global(action)
            next_state = process_map(new_state, env.height, env.width, simplify_agents=True)

            # Add experience and train
            for i, agent in enumerate(all_networks):
                agent.update_memory(state, action[i], reward, next_state, done)
                if agent.replay.count > agent.minimum_buffer:
                    agent.train()

            state = next_state
            total_reward += reward

            if total_reward <= -1000:
                break

        print("Total reward after episode {} is {}. Initialized as Done = {}. Done = {}."
              .format(episode, total_reward, env.initialized_done, done))

if __name__ == "__main__":
    run_individuals()