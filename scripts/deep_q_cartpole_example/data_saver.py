"""
From https://github.com/dennybritz/reinforcement-learning/blob/master/lib/plotting.py

Note: doesn't work great for cases where the total episodes is not pre-defined (like when updates are made step-by-step and not at the end)
"""

import matplotlib
import numpy as np
import pandas as pd
from collections import namedtuple
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import os

matplotlib.style.use('ggplot')

class DataSaver():
    def __init__(self, save_directory, total_episodes, version=0):
        self.save_dir = self.create_save_directory(save_directory, version)
        EpisodeStats = namedtuple("Stats",["episode_lengths", "episode_rewards"])
        self.stats = EpisodeStats(episode_lengths = np.zeros(total_episodes),
                                  episode_rewards = np.zeros(total_episodes))

    def create_save_directory(self, save_directory, version):
        sub_version = 0
        new_save_dir = save_directory + "V" + str(version) + "." + str(sub_version)

        while os.path.isdir(new_save_dir):
            sub_version += 1
            new_save_dir = save_directory + "V" + str(version) + "." + str(sub_version)

        os.mkdir(new_save_dir)
        return new_save_dir

    def plot_episode_stats(self, smoothing_window=10, noshow=False):
        # Plot the episode length over time
        fig1 = plt.figure(figsize=(10,5))
        plt.plot(self.stats.episode_lengths)
        plt.xlabel("Episode")
        plt.ylabel("Episode Length")
        plt.title("Episode Length over Time")
        if noshow:
            plt.close(fig1)
        else:
            plt.show(fig1)

        # Plot the episode reward over time
        fig2 = plt.figure(figsize=(10,5))
        rewards_smoothed = pd.Series(self.stats.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
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
        plt.plot(np.cumsum(self.stats.episode_lengths), np.arange(len(self.stats.episode_lengths)))
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


if __name__ == "__main__":
    ds = DataSaver("testing", 10)
#
#
# EpisodeStats = namedtuple("Stats",["episode_lengths", "episode_rewards"])
#
# def plot_cost_to_go_mountain_car(env, estimator, num_tiles=20):
#     x = np.linspace(env.observation_space.low[0], env.observation_space.high[0], num=num_tiles)
#     y = np.linspace(env.observation_space.low[1], env.observation_space.high[1], num=num_tiles)
#     X, Y = np.meshgrid(x, y)
#     Z = np.apply_along_axis(lambda _: -np.max(estimator.predict(_)), 2, np.dstack([X, Y]))
#
#     fig = plt.figure(figsize=(10, 5))
#     ax = fig.add_subplot(111, projection='3d')
#     surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
#                            cmap=matplotlib.cm.coolwarm, vmin=-1.0, vmax=1.0)
#     ax.set_xlabel('Position')
#     ax.set_ylabel('Velocity')
#     ax.set_zlabel('Value')
#     ax.set_title("Mountain \"Cost To Go\" Function")
#     fig.colorbar(surf)
#     plt.show()
#
#
# def plot_value_function(V, title="Value Function"):
#     """
#     Plots the value function as a surface plot.
#     """
#     min_x = min(k[0] for k in V.keys())
#     max_x = max(k[0] for k in V.keys())
#     min_y = min(k[1] for k in V.keys())
#     max_y = max(k[1] for k in V.keys())
#
#     x_range = np.arange(min_x, max_x + 1)
#     y_range = np.arange(min_y, max_y + 1)
#     X, Y = np.meshgrid(x_range, y_range)
#
#     # Find value for all (x, y) coordinates
#     Z_noace = np.apply_along_axis(lambda _: V[(_[0], _[1], False)], 2, np.dstack([X, Y]))
#     Z_ace = np.apply_along_axis(lambda _: V[(_[0], _[1], True)], 2, np.dstack([X, Y]))
#
#     def plot_surface(X, Y, Z, title):
#         fig = plt.figure(figsize=(20, 10))
#         ax = fig.add_subplot(111, projection='3d')
#         surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
#                                cmap=matplotlib.cm.coolwarm, vmin=-1.0, vmax=1.0)
#         ax.set_xlabel('Player Sum')
#         ax.set_ylabel('Dealer Showing')
#         ax.set_zlabel('Value')
#         ax.set_title(title)
#         ax.view_init(ax.elev, -120)
#         fig.colorbar(surf)
#         plt.show()
#
#     plot_surface(X, Y, Z_noace, "{} (No Usable Ace)".format(title))
#     plot_surface(X, Y, Z_ace, "{} (Usable Ace)".format(title))
#
#
#
# def plot_episode_stats(stats, smoothing_window=10, noshow=False):
#     # Plot the episode length over time
#     fig1 = plt.figure(figsize=(10,5))
#     plt.plot(stats.episode_lengths)
#     plt.xlabel("Episode")
#     plt.ylabel("Episode Length")
#     plt.title("Episode Length over Time")
#     if noshow:
#         plt.close(fig1)
#     else:
#         plt.show(fig1)
#
#     # Plot the episode reward over time
#     fig2 = plt.figure(figsize=(10,5))
#     rewards_smoothed = pd.Series(stats.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
#     plt.plot(rewards_smoothed)
#     plt.xlabel("Episode")
#     plt.ylabel("Episode Reward (Smoothed)")
#     plt.title("Episode Reward over Time (Smoothed over window size {})".format(smoothing_window))
#     if noshow:
#         plt.close(fig2)
#     else:
#         plt.show(fig2)
#
#     # Plot time steps and episode number
#     fig3 = plt.figure(figsize=(10,5))
#     plt.plot(np.cumsum(stats.episode_lengths), np.arange(len(stats.episode_lengths)))
#     plt.xlabel("Time Steps")
#     plt.ylabel("Episode")
#     plt.title("Episode per time step")
#     if noshow:
#         plt.close(fig3)
#     else:
#         plt.show(fig3)
#
#     return fig1, fig2, fig3
