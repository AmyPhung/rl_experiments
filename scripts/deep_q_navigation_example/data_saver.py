"""
From https://github.com/dennybritz/reinforcement-learning/blob/master/lib/plotting.py

Modifications:
- Uses lists instead of tuples, which allows for dynamic sizes of episodes
"""

import os
import time
import matplotlib
import gym
import numpy as np
import pandas as pd
from collections import namedtuple
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

matplotlib.style.use('ggplot')

class DataSaver():
    def __init__(self, env, model, save_directory, version=0):
        self.env = env
        self.model = model
        self.checkpoint_num = 0

        self.save_dir = self.create_save_directory(save_directory, version)
        self.episode_lengths = []
        self.episode_rewards = []
        self.start_time = time.time()

    def add_data_point(self, episode, length, reward):
        # Assumes episode numbers are indexed from 0
        if len(self.episode_lengths) <= episode:
            # Need to create new entry
            self.episode_lengths.append(length)
            self.episode_rewards.append(reward)
        else:
            # Modify existing data
            self.episode_lengths[episode] = length
            self.episode_rewards[episode] += reward

    def record_checkpoint(self):
        checkpoint_dir = self.save_dir + "/checkpoint" + str(self.checkpoint_num)
        # Save current model
        self.model.save(checkpoint_dir)

        # Render and save video for later viewing
        env = gym.wrappers.Monitor(self.env, checkpoint_dir, force=True)
        obs, done, ep_reward = env.reset(), False, 0
        while not done:
            action, q_values = self.model.action_value(obs[None])  # Using [None] to extend its dimension (4,) -> (1, 4)
            obs, reward, done, info = env.step(action)
            ep_reward += reward
        env.close()

        # Save current time and reward
        f = open(checkpoint_dir + "/notes.txt", "a")
        f.write("Elapsed time: " + str(time.time() - self.start_time))
        f.write("Test reward: " + str(ep_reward))
        f.close()

        self.checkpoint_num += 1

    def create_save_directory(self, save_directory, version):
        sub_version = 0
        new_save_dir = save_directory + "V" + str(version) + "." + str(sub_version)

        while os.path.isdir(new_save_dir):
            sub_version += 1
            new_save_dir = save_directory + "V" + str(version) + "." + str(sub_version)

        os.mkdir(new_save_dir)
        return new_save_dir

    def save_params_to_file(self, summary_string):
        f = open(self.save_dir + "/notes.txt", "a")
        f.write(summary_string)
        f.close()

    def plot_episode_stats(self, smoothing_window=1, noshow=False):
        # Plot the episode length over time
        fig1 = plt.figure(figsize=(10,5))
        plt.plot(self.episode_lengths)
        plt.xlabel("Episode")
        plt.ylabel("Episode Length")
        plt.title("Episode Length over Time")
        if noshow:
            plt.close(fig1)
        else:
            plt.show(fig1)

        # Plot the episode reward over time
        fig2 = plt.figure(figsize=(10,5))
        rewards_smoothed = pd.Series(self.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
        plt.plot(rewards_smoothed)
        plt.xlabel("Episode")
        plt.ylabel("Episode Reward (Smoothed)")
        plt.title("Episode Reward over Time (Smoothed over window size {})".format(smoothing_window))
        if noshow:
            plt.close(fig2)
        else:
            plt.show(fig2)

        # Plot time steps and episode number
        fig3 = plt.figure(figsize=(10,5))
        plt.plot(np.cumsum(self.episode_lengths), np.arange(len(self.episode_lengths)))
        plt.xlabel("Time Steps")
        plt.ylabel("Episode")
        plt.title("Episode per time step")
        if noshow:
            plt.close(fig3)
        else:
            plt.show(fig3)

        return fig1, fig2, fig3

    def plot_and_save_stats(self):
        fig1, fig2, fig3 = self.plot_episode_stats()
        fig1.savefig(self.save_dir + "/length_over_time")
        fig2.savefig(self.save_dir + "/reward_over_time")
        fig3.savefig(self.save_dir + "/episode_per_timestep")
        data = np.array([self.episode_lengths, self.episode_rewards]).T
        df = pd.DataFrame(data, columns =['Episode Lengths', 'Episode Rewards'])
        df.to_csv(self.save_dir + "/training_stats.csv")


if __name__ == "__main__":
    ds = DataSaver("testing", 10)
