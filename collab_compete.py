import os
import sys
import argparse
from collections import deque

import torch
import numpy as np
import matplotlib.pyplot as plt
from unityagents import UnityEnvironment

from ddpg_agent import Agent, ReplayBuffer

# environment configuration
env = UnityEnvironment(file_name="Tennis_Linux/Tennis.x86_64")

brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents
num_agents = len(env_info.agents)
print('Number of agents:', num_agents)

# size of each action
action_size = brain.vector_action_space_size
print('Size of each action:', action_size)

# examine the state space
states = env_info.vector_observations
state_size = states.shape[1]
print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
print('The state for the first agent looks like:', states[0])


def save(agents):
    for i, agent in enumerate(agents):
        torch.save(agent.actor_local.state_dict(), 'checkpoint_actor{}.pth'.format(i))
        torch.save(agent.critic_local.state_dict(), 'checkpoint_critic{}.pth'.format(i))


def load(agents):
    for i, agent in enumerate(agents):
        if os.path.isfile('checkpoint_actor{}.pth'.format(i)) and os.path.isfile('checkpoint_critic{}.pth'.format(i)):
            agent.actor_local.load_state_dict(torch.load('checkpoint_actor{}.pth'.format(i)))
            agent.actor_target.load_state_dict(torch.load('checkpoint_actor{}.pth'.format(i)))
            agent.critic_local.load_state_dict(torch.load('checkpoint_critic{}.pth'.format(i)))
            agent.critic_target.load_state_dict(torch.load('checkpoint_critic{}.pth'.format(i)))


def ddpg_train(n_episodes, seed, buffer_size, batch_size, gamma, tau, lr_actor, lr_critic, weight_decay):
    memory = ReplayBuffer(action_size, buffer_size, batch_size, seed)
    agents = [Agent(state_size, action_size, seed, buffer_size, batch_size, gamma, tau, lr_actor, lr_critic,
                    weight_decay, memory) for _ in range(num_agents)]
    load(agents)
    scores_deque = deque(maxlen=100)
    scores = []

    for i_episode in range(1, n_episodes + 1):
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations
        episode_scores = np.zeros(num_agents)
        while True:
            for agent in agents:
                agent.reset()
            actions = list()
            for agent, state in zip(agents, states):
                actions.append(agent.act(state))
            env_info = env.step(actions)[brain_name]
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done
            for agent, state, action, reward, next_state, done in zip(agents, states, actions, rewards, next_states,
                                                                      dones):
                agent.step(state, action, reward, next_state, done)
            states = next_states
            episode_scores += np.array(rewards)
            if np.any(dones):
                break
        score = episode_scores.max()
        scores_deque.append(score)
        scores.append(score)
        print('\rEpisode: \t{} \tScore: \t{:.2f} \tAverage Score: \t{:.2f}'.format(
            i_episode, np.mean(score), np.mean(scores_deque)), end="")
        if i_episode % 10 == 0:
            save(agents)
        if np.mean(scores_deque) >= 0.5:
            print(
                '\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode,
                                                                                       np.mean(scores_deque)))
            break
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.grid()
    ax.plot(np.arange(len(scores)), scores)
    ax.set(xlabel="Episode #", ylabel="'Score", title="DDPG Network")
    fig.savefig("ddpg_network.pdf")


def ddpg_test():
    agents = [Agent(state_size, action_size, 0, 0, 0, 0, 0, 0, 0,
                    0, 0) for _ in range(num_agents)]
    load(agents)
    for i_episode in range(3):
        env_info = env.reset(train_mode=False)[brain_name]
        states = env_info.vector_observations
        while True:
            for agent in agents:
                agent.reset()
            actions = list()
            for agent, state in zip(agents, states):
                actions.append(agent.act(state))
            env_info = env.step(actions)[brain_name]
            next_states = env_info.vector_observations
            dones = env_info.local_done
            states = next_states
            if np.any(dones):
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Udacity Deep Reinforcement Learning Nano Degree - Project 2 Continuous Control')
    parser.add_argument('--n_episodes', metavar='', type=int,
                        default=1000, help='maximum number of training episodes')
    parser.add_argument('--seed', metavar='', type=int,
                        default=0, help='seed for stochastic variables')
    parser.add_argument('--buffer_size', metavar='', type=int,
                        default=int(1e5), help='replay buffer size')
    parser.add_argument('--batch_size', metavar='', type=int,
                        default=64, help='minibatch size')
    parser.add_argument('--gamma', metavar='', type=float,
                        default=0.99, help='discount factor')
    parser.add_argument('--tau', metavar='', type=float,
                        default=1e-3, help='for soft update of target parameters')
    parser.add_argument('--lr_actor', metavar='', type=float,
                        default=5e-4, help='learning rate for actor')
    parser.add_argument('--lr_critic', metavar='', type=float,
                        default=5e-4, help='learning rate for agent')
    parser.add_argument('--weight_decay', metavar='', type=int,
                        default=0, help='L2 weight decay')
    parser.add_argument('--train_test', metavar='', type=int,
                        default=0, help='0 to train and 1 to test agent')
    args = parser.parse_args()

    if args.train_test == 0:
        ddpg_train(args.n_episodes, args.seed, args.buffer_size,
                   args.batch_size, args.gamma, args.tau, args.lr_actor,
                   args.lr_critic, args.weight_decay)
    elif args.train_test == 1:
        ddpg_test()
    else:
        print("invalid argument for train_test, please use 0 to train and 1 to test agent")
        sys.exit(1)
