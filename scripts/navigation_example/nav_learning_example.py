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

from nav_env import NavEnv
import plotting
from scores.score_logger import ScoreLogger

ENV_NAME = 'navigationV0.4'

class NavigationQAgent():
    def __init__(self,
                 obs_buckets=(4, 4, 6, 4, 4),
                 act_buckets=(3, 3),
                 num_episodes=100000,
                 min_lr=0.1,
                 min_epsilon=0.1,
                 discount=1.0,
                 decay=25):

        self.obs_buckets = obs_buckets
        self.act_buckets = act_buckets
        self.num_episodes = num_episodes
        self.min_lr = min_lr
        self.min_epsilon = min_epsilon
        self.discount = discount
        self.decay = decay

        self.env = NavEnv()

        # [position, velocity, angle, angular velocity]
        # State: robot-x, robot-y, robot-theta, goal-x, goal-y
        self.state_upper_bounds = self.env.observation_space.high
        self.state_lower_bounds = self.env.observation_space.low

        self.action_upper_bounds = self.env.action_space.high
        self.action_lower_bounds = self.env.action_space.low

        self.Q_table = np.zeros(self.obs_buckets + self.act_buckets, dtype=np.float32)

    def discretize(self, input, num_buckets, lower_bounds, upper_bounds):
        """ Used to discretize state or action.

        Args:
            input (tuple or list): either the action or the state
            num_buckets (tuple or list): number of buckets for corresponding input
            lower_bounds (tuple or list): lower bounds for corresponding input
            upper_bounds (tuple or list): upper bounds for corresponding input
        """
        discretized = list()
        for i in range(len(input)):
            if input[i] >= upper_bounds[i]:
                # Save max idx if input is higher than expected
                discretized.append(num_buckets[i] - 1)
            elif input[i] <= lower_bounds[i]:
                # Save min idx (0) if input is lower than expected
                discretized.append(0)
            else:
                # Obs is within expected range
                input_range_size = (upper_bounds[i] - lower_bounds[i]) / num_buckets[i]
                new_input = int((input[i] - lower_bounds[i]) // input_range_size)
                discretized.append(new_input)
        return tuple(discretized)

    def discretize_state(self, state):
        return self.discretize(state, self.obs_buckets, self.state_lower_bounds,
                               self.state_upper_bounds)

    def discretize_action(self, action):
        return self.discretize(action, self.act_buckets, self.action_lower_bounds,
                               self.action_upper_bounds)

    def undiscretize_action(self, action):
        undiscretized = list()
        for i in range(len(action)):
            range_size = (self.action_upper_bounds[i] - self.action_lower_bounds[i]) / self.act_buckets[i]
            # Compute center of range for action
            new_action = range_size*action[i] + self.action_lower_bounds[i] + range_size/2
            undiscretized.append(new_action)
        return tuple(undiscretized)

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
            max_nonzero_val = np.max(self.Q_table[state][nonzero_idxs], axis=None)
            idx = np.argwhere(self.Q_table[state] == max_nonzero_val)

            # Convert index to valid action
            action = self.undiscretize_action(idx[0])
            return np.float32(action)

    def update_q(self, state, action, reward, new_state):
        self.Q_table[state][action] += self.learning_rate * (reward + self.discount * np.max(self.Q_table[new_state]) - self.Q_table[state][action])

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
            current_state = self.discretize_state(self.env.reset())

            self.learning_rate = self.get_learning_rate(e)
            self.epsilon = self.get_epsilon(e)
            t = 0
            done = False

            while not done:
                t = t+1
                action = self.choose_action(current_state)
                obs, reward, done, _ = self.env.step(action)
                new_state = self.discretize_state(obs)
                new_action = self.discretize_action(action)
                self.update_q(current_state, new_action, reward, new_state)
                current_state = new_state

                stats.episode_rewards[e] += reward
                stats.episode_lengths[e] = t

                if done:
                    score_logger.add_score(stats.episode_rewards[e], e)

                if e%100 == 0:
                    # Render every 10 episodes
                    # self.env.render()
                    # Print stats every 10 episodes
                    if done:
                        print("Episode: " + str(e))
                        print("Rewards: " + str(stats.episode_rewards[e]))

        print('Finished training!')
        return stats

    def run(self):
        self.env = gym.wrappers.Monitor(self.env,ENV_NAME,force=True)
        t = 0
        done = False
        current_state = self.discretize_state(self.env.reset())

        while not done:
                self.env.render()
                t = t+1
                action = self.choose_action(current_state)
                obs, reward, done, _ = self.env.step(action)
                new_state = self.discretize_state(obs)
                current_state = new_state
        return t



if __name__ == "__main__":
    agent = NavigationQAgent()
    stats = agent.train()
    t = agent.run()
    print("Time", t)
    plotting.plot_episode_stats(stats)
