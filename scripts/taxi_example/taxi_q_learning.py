"""
https://thomassimonini.medium.com/q-learning-lets-create-an-autonomous-taxi-part-2-2-8cbafa19d7f5
"""

import numpy as np
import gym
import random

class TaxiQAgent():
    def __init__(self):
        # Create environment
        self.env = gym.make("Taxi-v3")
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
                # If done : finish episode
                if done == True:
                    break

                # Our new state is state
                state = new_state


if __name__ == "__main__":
    taxi_q_agent = TaxiQAgent()
    taxi_q_agent.env.render()

    taxi_q_agent.train()
