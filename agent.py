import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, n_latent_var):
        super(ActorCritic, self).__init__()

        # actor
        self.action_layer = nn.Sequential(
            nn.Linear(state_dim, n_latent_var),
            nn.Tanh(),
            nn.Linear(n_latent_var, n_latent_var),
            nn.Tanh(),
            nn.Linear(n_latent_var, action_dim),
            nn.Softmax(dim=-1)
        )

        # critic
        self.value_layer = nn.Sequential(
            nn.Linear(state_dim, n_latent_var),
            nn.Tanh(),
            nn.Linear(n_latent_var, n_latent_var),
            nn.Tanh(),
            nn.Linear(n_latent_var, 1)
        )

    def forward(self):
        raise NotImplementedError

    def act(self, state, memory):
        state = torch.from_numpy(state).float().to(device)
        action_probs = self.action_layer(state)
        dist = Categorical(action_probs)
        action = dist.sample()

        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(dist.log_prob(action))

        return action.item()

    def evaluate(self, state, action):
        action_probs = self.action_layer(state)
        dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()

        state_value = self.value_layer(state)

        return action_logprobs, torch.squeeze(state_value), dist_entropy


class PPO:
    def __init__(self, state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.policy = ActorCritic(state_dim, action_dim, n_latent_var).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)
        self.policy_old = ActorCritic(state_dim, action_dim, n_latent_var).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def update(self, memory):
        # Monte Carlo estimate of state rewards:
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards:
        rewards = torch.tensor(rewards).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        # convert list to tensor
        old_states = torch.stack(memory.states).to(device).detach()
        old_actions = torch.stack(memory.actions).to(device).detach()
        old_logprobs = torch.stack(memory.logprobs).to(device).detach()

        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            # Evaluating old actions and values :
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss:
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())


class RandomAgent(object):
    """The world's simplest agent!"""

    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, *args):
        return np.random.choice(self.action_space)

    def reinitialise(self):
        pass


class Dyna_Q(object):

    def __init__(self, action_space):
        self.action_space = action_space

        print(action_space)

        self.alpha = 0.1
        self.alpha_r = 0.5
        self.alpha_p = 0.5

        self.gamma = 0.95
        self.eps = 1
        self.last_obs = None
        self.last_a = None
        self.Q = dict()
        self.R = dict()
        self.P = dict()

    def act(self, obs, reward, done):
        obs = np.array_str(obs)
        if obs not in self.Q.keys():
            self.Q[obs] = [0 for _ in range(self.action_space)]

        if (self.last_obs, self.last_a, obs) not in self.R.keys():
            self.R[(self.last_obs, self.last_a, obs)] = 0
            self.P[(self.last_obs, self.last_a, obs)] = 0

        if np.random.random_sample() > self.eps:
            a = self.Q[obs].index(max(self.Q[obs]))
        else:
            a = np.random.choice(self.action_space)

        self.update_Q(reward, obs)
        self.update_R(reward, obs)
        self.update_P(reward, obs)

        self.last_obs = obs
        self.last_a = a

        if done:
            self.eps *= 0.9999
            self.alpha_r *= 0.9993
            self.alpha_p *= 0.9993

        return a

    def update_Q(self, reward, obs):

        if self.last_obs == None or self.last_a == None:
            return

        a = self.Q[obs].index(max(self.Q[obs]))

        self.Q[self.last_obs][self.last_a] += self.alpha * (
                reward + self.gamma * self.Q[obs][a] - self.Q[self.last_obs][self.last_a])

    def update_R(self, reward, obs):

        if self.last_obs == None or self.last_a == None:
            return

        self.R[(self.last_obs, self.last_a, obs)] += self.alpha_r * (reward - self.R[(self.last_obs, self.last_a, obs)])

    def update_P(self, reward, obs):

        if self.last_obs is None or self.last_a is None:
            return

        self.P[(self.last_obs, self.last_a, obs)] += self.alpha_p * (reward - self.P[(self.last_obs, self.last_a, obs)])

        for triple in self.P:
            if triple[1] == self.last_a and triple[0] == self.last_obs:
                if triple[2] == obs:
                    pass
                self.P[triple] += self.alpha_p * (-self.P[triple])

    def reinitialise(self):
        self.last_obs = None
        self.last_a = None


class Sarsa:
    def __init__(self, action_space):
        self.action_space = action_space

        self.alpha = 0.1
        self.gamma = 0.95
        self.eps = 1
        self.last_obs = None
        self.last_a = None
        self.Q = dict()

    def act(self, obs, reward, done):
        obs = np.array_str(obs)
        if obs not in self.Q.keys():
            self.Q[obs] = [0 for _ in range(self.action_space)]

        if np.random.random_sample() > self.eps:
            a = self.Q[obs].index(max(self.Q[obs]))
        else:
            a = np.random.choice(self.action_space)
        self.update_Q(reward, obs, a)

        self.last_obs = obs
        self.last_a = a

        if done:
            self.eps *= 0.996

        return a

    def update_Q(self, reward, obs, a):

        if self.last_obs == None or self.last_a == None:
            return

        self.Q[self.last_obs][self.last_a] += self.alpha * (
                reward + self.gamma * self.Q[obs][a] - self.Q[self.last_obs][self.last_a])

    def reinitialise(self):
        self.last_obs = None
        self.last_a = None

