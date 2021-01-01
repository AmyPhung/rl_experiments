"""
https://thomassimonini.medium.com/q-learning-lets-create-an-autonomous-taxi-part-2-2-8cbafa19d7f5

500 states
    5x5 grid creates...
        25 places for taxi (unloaded)
            *4 possible starting locations for passenger
            *4 possible goal locations
        = 400
        25 places for taxi (loaded)
            *4 possible goal locations
        = 100
6 actions (up/down/left/right, pickup, dropoff)

Trains in ~30 sec
"""

import numpy as np
import gym
import random
from data_saver import DataSaver
import time

ENV_NAME = "Taxi-v3"

class TaxiQAgent():
    def __init__(self):
        # Create environment
        self.env = gym.make(ENV_NAME)
        self.state_space = self.env.observation_space.n
        self.action_space = self.env.action_space.n

        # Initialize Q table
        self.Q = np.zeros((self.state_space, self.action_space))

        # Define hyperparameters
        self.total_episodes = 25000        # Total number of training episodes
        self.total_test_episodes = 100     # Total number of test episodes
        self.max_steps = 200               # Max steps per episode

        self.learning_rate = 0.01          # Learning rate
        self.gamma = 0.99                  # Discounting rate

        # Exploration parameters
        self.epsilon = 1.0                 # Exploration rate
        self.max_epsilon = 1.0             # Exploration probability at start
        self.min_epsilon = 0.001           # Minimum exploration probability
        self.decay_rate = 0.01             # Exponential decay rate for exploration prob

        self.data_saver = DataSaver(ENV_NAME, self.total_episodes)

    def epsilon_greedy_policy(self, state):
        # if random number > greater than epsilon --> exploitation
        if(random.uniform(0,1) > self.epsilon):
            action = np.argmax(self.Q[state])
        # else --> exploration
        else:
            action = self.env.action_space.sample()

        return action

    def get_epsilon(self, e):
        return self.min_epsilon + (self.max_epsilon - self.min_epsilon)*np.exp(-self.decay_rate*e)

    def train(self):
        for episode in range(self.total_episodes):
            # Reset the environment
            state = self.env.reset()
            step = 0
            done = False

            # Reduce epsilon (because we need less and less exploration)
            self.epsilon = self.get_epsilon(episode)

            for step in range(self.max_steps):
                action = self.epsilon_greedy_policy(state)

                # Take the action (a) and observe the outcome state(s') and reward (r)
                new_state, reward, done, info = self.env.step(action)

                # Update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
                self.Q[state][action] = self.Q[state][action] + self.learning_rate * (reward + self.gamma *
                                            np.max(self.Q[new_state]) - self.Q[state][action])

                # Save stats
                # print(self.data_saver.stats.episode_rewards)
                self.data_saver.stats.episode_rewards[episode] += reward
                self.data_saver.stats.episode_lengths[episode] = step

                # If done : finish episode
                if done == True:
                    # score_logger.add_score(stats.episode_rewards[e], e)
                    break

                # Our new state is state
                state = new_state

        print("Finished Training!")
        self.data_saver.plot_and_save_stats()

    def test(self):
        self.env = gym.wrappers.Monitor(self.env, self.data_saver.save_dir, force=True)
        t = 0
        done = False
        current_state = self.env.reset()

        while not done:
            self.env.render()
            t = t+1
            action = self.epsilon_greedy_policy(current_state)
            new_state, reward, done, _ = self.env.step(action)
            current_state = new_state
        return t


if __name__ == "__main__":
    taxi_q_agent = TaxiQAgent()
    taxi_q_agent.env.render()

    t_start = time.time()
    taxi_q_agent.train()
    t_end = time.time()
    print ("Time elapsed: " + str(t_end - t_start))

    taxi_q_agent.test()
    taxi_q_agent.env.close()
