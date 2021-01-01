"""
Based on https://github.com/IsaacPatole/CartPole-v0-using-Q-learning-SARSA-and-DNN/blob/master/Qlearning_for_cartpole.py

Modified to use plotting, new navigation environment
Discretized to reduce training speeds at the cost of performance

We will discretize actions
backwards left
left only
forwards left
backwards straight
stop
forwards straight
forwards right
right only
backwards right
"""

import gym
import numpy as np
import math
import time

from nav_env import NavEnv
import plotting
from scores.score_logger import ScoreLogger

import tkinter
import matplotlib
matplotlib.use('TkAgg')
matplotlib.style.use('ggplot')

ENV_NAME = 'discrete_navigationV0.1'

class NavigationQAgent():
    def __init__(self,
                 num_episodes=50000,
                 min_lr=0.1,
                 min_epsilon=0.1,
                 discount=1.0,
                 decay=25):

        self.env = NavEnv()

        self.num_episodes = num_episodes
        self.min_lr = min_lr
        self.min_epsilon = min_epsilon
        self.discount = discount
        self.decay = decay

        # State: robot-x, robot-y, robot-direction, goal-x, goal-y
        obs_buckets = (self.env.world_x_limit,
                       self.env.world_y_limit,
                       4,
                       self.env.world_x_limit,
                       self.env.world_y_limit)

        # Straight, right, left
        act_buckets = self.env.action_space.n

        self.Q_table = np.zeros(obs_buckets + (act_buckets,), dtype=np.float32)

    def choose_action(self, state):
        if (np.random.random() < self.epsilon):
            # Explore based on the current value of the epsilon
            return self.env.action_space.sample()

        else:
            # Refer to Q table the other % of the time
            nonzero_idxs = np.nonzero(self.Q_table[state])

            # If the state has not been explored yet, choose a random action
            if len(nonzero_idxs[0]) == 0:
                return self.env.action_space.sample()

            # Get index of highest saved non-zero Q value
            action = np.argmax(nonzero_idxs)
            return action

    def update_q(self, state, action, reward, new_state):
        r_x, r_y, direction, g_x, g_y = new_state

        outside_limit = bool(
            r_x < 0
            or r_x >= self.env.world_x_limit
            or r_y < 0
            or r_y >= self.env.world_y_limit)

        if outside_limit:
            future_reward = self.env.exit_reward
        else:
            future_reward = np.max(self.Q_table[new_state])

        self.Q_table[state][action] += self.learning_rate * (reward + self.discount * future_reward - self.Q_table[state][action])
        print("HERE")
        print(self.Q_table[state][action])
        # time.sleep(1)

    def get_epsilon(self, t):
        return max(self.min_epsilon, min(1., 1. - math.log10((t + 1) / self.decay)))

    def get_learning_rate(self, t):
        return max(self.min_lr, min(1., 1. - math.log10((t + 1) / self.decay)))

    def train(self):
        stats = plotting.EpisodeStats(
            episode_lengths = np.zeros(self.num_episodes),
            episode_rewards = np.zeros(self.num_episodes))

        score_logger = ScoreLogger(ENV_NAME)

        for e in range(self.num_episodes):
            current_state = self.env.reset()

            self.learning_rate = self.get_learning_rate(e)
            self.epsilon = self.get_epsilon(e)
            t = 0
            done = False

            while not done:
                t = t+1
                action = self.choose_action(current_state)
                new_state, reward, done, _ = self.env.step(action)
                self.update_q(current_state, action, reward, new_state)

                stats.episode_rewards[e] += reward
                stats.episode_lengths[e] = t

                if done:
                    score_logger.add_score(stats.episode_rewards[e], e)


                #debugging
                # if e == 100:
                # self.env.render()
                # time.sleep(1)
                # print(reward)

        # #
        #         if e%100 == 0:
        #             self.env.render()
        #             if done:
        #                 print("Learning Rate: " + str(self.learning_rate))
        #                 print("Epsilon: " + str(self.epsilon))
        #                 print("Episode: " + str(e))
        #                 print("Rewards: " + str(stats.episode_rewards[e]))
        #         if e>=10000 and e%100 == 0:
        #             # Render every 100 episodes
        #             self.env.render()
        #             # Print stats every 100 episodes
        #             if done:
        #                 print("Episode: " + str(e))
        #                 print("Rewards: " + str(stats.episode_rewards[e]))
        #
        print('Finished training!')
        return stats

    def run(self):
        self.env = gym.wrappers.Monitor(self.env,ENV_NAME,force=True)
        t = 0
        done = False
        current_state = self.env.reset()

        while not done:
                self.env.render()
                t = t+1
                action = self.choose_action(current_state)
                new_state, reward, done, _ = self.env.step(action)
                current_state = new_state
        return t



if __name__ == "__main__":
    agent = NavigationQAgent()
    stats = agent.train()
    t = agent.run()
    print("Time", t)
    plotting.plot_episode_stats(stats)
