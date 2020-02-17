from agent import *
from my_env import *


def main_ppo(env, episodes):

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    n_latent_var = 64
    lr = 0.002
    betas = (0.9, 0.999)
    gamma = 0.99  # discount factor
    K_epochs = 4  # update policy for K epochs
    eps_clip = 0.2  # clip parameter for PPO

    memory = Memory()
    agent = PPO(state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip)

    timestep = 0
    update_timestep = 2000

    res_reward, res_nbaction = [], []

    for episode in range(episodes):
        print("Episode: %s, FPS: %s, UPS: %s" % (episode, env.game.get_fps(), env.game.get_ups()))

        terminal = False
        state = env.reset()

        nb_step = 0
        sum_reward = 0
        while not terminal and nb_step < 2000:

            action = agent.policy_old.act(state, memory)
            next_state, reward, terminal, _ = env.step(action)  # .data.item()

            memory.rewards.append(reward)
            memory.is_terminals.append(terminal)

            timestep += 1
            if timestep % update_timestep == 0:
                agent.update(memory)
                memory.clear_memory()
                timestep = 0

            nb_step += 1
            sum_reward += reward

        res_reward.append(sum_reward)
        res_nbaction.append(nb_step)

        print(sum_reward, nb_step)

    return res_reward, res_nbaction


def main_dynaq(env, episodes, agent):

    res_reward, res_nbaction = [], []

    for episode in range(episodes):
        print("Episode: %s, FPS: %s, UPS: %s" % (episode, env.game.get_fps(), env.game.get_ups()))

        terminal = False
        state = env.reset()

        nb_step = 0
        sum_reward = 0
        reward = 0

        while not terminal and nb_step < 30_000:

            action = agent.act(state, reward, terminal)
            state, reward, terminal, _ = env.step(action)  # .data.item()

            nb_step += 1
            sum_reward += reward

            if terminal:
                agent.act(state, reward, terminal)
                agent.reinitialise()

        res_reward.append(sum_reward)
        res_nbaction.append(nb_step)
        print(sum_reward, nb_step)
    return res_reward, res_nbaction


if __name__ == "__main__":
    episodes = 1000

    env = GoldCollect({}, 100, 'gold')  # ou lumber pour le bois

    env.game.set_max_fps(100000)
    env.game.set_max_ups(100000)


    #Optimal
    print('score optimal =', env.calculate_optimal_play())  # total_steps, total_reward


    # PPO
    print(main_ppo(env, episodes))


    # Dyna-Q, Sarsa et random
    action_dim = env.action_space.n
    #agent = Dyna_Q(action_dim)
    #agent = RandomAgent(action_dim)
    #agent = Sarsa(action_dim)

    #print(main_dynaq(env, episodes, agent))



