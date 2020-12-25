"""
Simple system where a robot attempts to navigate towards a goal.
Based on cartpole example

Helpful Inpsiration:
Car racing example: https://github.com/openai/gym/blob/master/gym/envs/box2d/car_racing.py
"""

import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np


class NavEnv(gym.Env):
    """
    Description:
        A circular robot exists in an obstacle-free environment and can turn
        or drive forwards. The robot and goal states are randomized, and the
        goal is to drive the robot to the goal state.

    Observation:
        Type: Box(5)
        Num     Observation               Min                Max
        0       Robot X-Position          -sim x-limit       sim x-limit
        1       Robot Y-Position          -sim y-limit       sim y-limit
        2       Robot direction           -pi                pi
        3       Goal X-Position           -sim x-limit       sim x-limit
        4       Goal Y-Position           -sim y-limit       sim y-limit

    Actions:
        Type: Box(2)
        Num   Action                      Min           Max
        0     Linear velocity (m/s)       -0.5          0.5
        1     Steering (rad/s)            -3            3

    Reward:
        Reward is -0.7 for every time step, +100000 for reaching the goal, and
        -100000 for leaving the map, +0.3 * traversed distance to goal in the
        last timestep

    Starting State:
        All observations are assigned a uniform random value in [-0.05..0.05]

    Episode Termination:
        Pole Angle is more than 12 degrees.
        Cart Position is more than 2.4 (center of the cart reaches the edge of
        the display).
        Episode length is greater than 200.
        Solved Requirements:
        Considered solved when the average return is greater than or equal to
        195.0 over 100 consecutive trials.
    """

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self):
        # Environment parameters
        self.robot_diameter = 0.3 # meters
        self.goal_diameter = 1.0 # meters
        self.max_linear_vel = 0.4 # meters/sec
        self.max_angular_vel = 5 # radians/sec

        # Rewards
        self.goal_reward = 100000
        self.exit_reward = -100000
        self.time_reward = -0.7
        self.distance_reward = 0.3

        # Distance at which to fail the episode
        self.world_x_limit = 3
        self.world_y_limit = 2

        # State update parameters
        self.tau = 0.02  # seconds between state updates
        self.kinematics_integrator = 'euler'

        # Meters to pixels conversion for render
        self.scale = 150

        # Set constraints on action and observation spaces
        action_lim = np.array([self.max_linear_vel,
                               self.max_angular_vel],
                              dtype=np.float32)
        obs_lim = np.array([self.world_x_limit,
                            self.world_y_limit,
                            np.pi,
                            self.world_x_limit,
                            self.world_y_limit],
                           dtype=np.float32)
        self.action_space = spaces.Box(-action_lim, action_lim, dtype=np.float32)
        self.observation_space = spaces.Box(-obs_lim, obs_lim, dtype=np.float32)

        self.seed()
        self.viewer = None
        self.state = None
        self.steps_beyond_done = None
        self.prev_dist = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        # Update state ---------------------------------------------------------
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg

        r_x, r_y, r_theta, g_x, g_y = self.state
        lin_vel, ang_vel = action

        r_theta += self.tau * ang_vel
        # Keep theta within -pi to pi range
        r_theta = r_theta % (2*np.pi)
        if r_theta > np.pi:
            r_theta -= 2*np.pi

        r_x += lin_vel*np.cos(r_theta)
        r_y += lin_vel*np.sin(r_theta)

        self.state = (r_x, r_y, r_theta, g_x, g_y)

        # Update reward --------------------------------------------------------
        # Check if robot is within goal
        curr_dist = np.linalg.norm([r_x-g_x, r_y-g_y])
        within_goal = bool(curr_dist < (self.goal_diameter - self.robot_diameter))

        # Check if we left the field
        outside_limit = bool(
            r_x < -self.world_x_limit
            or r_x > self.world_x_limit
            or r_y < -self.world_x_limit
            or r_y > self.world_x_limit)

        # Compute rewards
        if not within_goal and not outside_limit:
            # We're not done yet - neither terminating case has been reached
            done = False
            dist_delta = self.prev_dist - curr_dist
            reward = self.time_reward + (dist_delta * self.distance_reward)

        elif self.steps_beyond_done is None:
            # We just reached a terminating case
            done = True
            if outside_limit:
                reward = self.exit_reward
                self.steps_beyond_done = 0

            elif within_goal:
                reward = self.goal_reward
                self.steps_beyond_done = 0
        else:
            # We've reached a terminating case again
            done = True
            if self.steps_beyond_done == 0:
                logger.warn(
                    "You are calling 'step()' even though this "
                    "environment has already returned done = True. You "
                    "should always call 'reset()' once you receive 'done = "
                    "True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_done += 1
            reward = 0.0

        return np.array(self.state), reward, done, {}

    def reset(self):
        # State: robot-x, robot-y, robot-theta, goal-x, goal-y
        self.state = [self.np_random.uniform(low=-self.world_x_limit,
                                             high=self.world_x_limit),
                      self.np_random.uniform(low=-self.world_y_limit,
                                             high=self.world_y_limit),
                      self.np_random.uniform(low=-np.pi,
                                             high=np.pi),
                      self.np_random.uniform(low=-self.world_x_limit,
                                             high=self.world_x_limit),
                      self.np_random.uniform(low=-self.world_y_limit,
                                             high=self.world_y_limit)]
        self.steps_beyond_done = None
        r_x, r_y, r_theta, g_x, g_y = self.state
        self.prev_dist = np.linalg.norm([r_x-g_x, r_y-g_y])
        return np.array(self.state)

    def render(self, mode='human'):
        screen_width = int(self.scale * 2*self.world_x_limit)
        screen_height = int(self.scale * 2*self.world_y_limit)

        world_width = self.world_x_limit * 2
        world_height = self.world_y_limit * 2

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            robot = rendering.make_circle(self.scale * self.robot_diameter)
            self.robottrans = rendering.Transform()
            robot.add_attr(self.robottrans)
            robot.set_color(0.0, 0.1, 0.5)
            self.viewer.add_geom(robot)

            direction = rendering.make_polyline([(0, 0),
                (self.scale * self.robot_diameter, 0)])
            self.directiontrans = rendering.Transform()
            direction.add_attr(self.directiontrans)
            direction.set_color(0.8, 0.8, 0.8)
            self.viewer.add_geom(direction)

            goal = rendering.make_circle(self.scale * self.goal_diameter)
            self.goaltrans = rendering.Transform()
            goal.add_attr(self.goaltrans)
            goal.set_color(0.1, 0.5, 0.1)
            self.viewer.add_geom(goal)

        if self.state is None:
            return None

        x = self.state

        robotx = (x[0] + self.world_x_limit) * self.scale
        roboty = (x[1] + self.world_y_limit) * self.scale
        goalx = (x[3] + self.world_x_limit) * self.scale
        goaly = (x[4] + self.world_y_limit) * self.scale
        self.robottrans.set_translation(robotx, roboty)
        self.directiontrans.set_translation(robotx, roboty)
        self.directiontrans.set_rotation(x[2])
        self.goaltrans.set_translation(goalx, goaly)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

if __name__ == "__main__":
    # For testing step functions
    import time
    nav_env = NavEnv()
    nav_env.reset()

    time.sleep(1)
    sample_action = nav_env.action_space.sample()

    while True:
        nav_env.render()
        state, reward, done, _ = nav_env.step(sample_action)
        print(reward)
        if done:
            while True:
                pass
