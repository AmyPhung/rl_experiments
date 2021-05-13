"""
Simple system where a robot attempts to navigate towards a goal.
Based on cartpole example

Helpful Inpsiration:
Car racing example: https://github.com/openai/gym/blob/master/gym/envs/box2d/car_racing.py

Discrete actions, continuous observation space

TODO: maybe we need a reward for turning towards the goal?
TODO: Make reward scaling more correct
TODO: Make sure the robot won't spawn in the goal
TODO: Make new text logging, record training time
(put this in a file self, model, target_model, env, buffer_size=100, learning_rate=.0015, epsilon=.1, epsilon_dacay=0.995,
             min_epsilon=.01, gamma=.95, batch_size=4, target_update_iter=400, train_nums=50000, start_learning=10)
TODO: Make stats save every few episodes instead of all at once, save actual model

Things to try:
- fix bug in angles at edge (wrapping at 0)
- add cost to staying still
- keeping spawn locations fixed
Notes: Our hand-coded "reward function" is the combination of distance & heading. Our RL model will attempt to learn this (in theory, if we just gave our RL model these inputs, it should be a lot easier)
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
        Type: Discrete(3)
        Num   Action
        0     Move straight (0.4 meters/sec)
        1     Turn left (5 radians/sec)
        2     Turn right (-5 radians/sec)

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
        self.max_linear_vel = 0.5 # meters/sec
        self.max_angular_vel = 3 # radians/sec

        # Rewards
        self.goal_reward = 1000
        self.exit_reward = -100
        self.time_reward = -1
        self.distance_reward = 12 # Avg distance magnitude ~0.01
        # self.angle_reward = 0 # Avg angle magnitude ~0.06
        self.angle_reward = 1.5 # Avg angle magnitude ~0.06

        # Distance at which to fail the episode
        self.world_x_limit = 2
        self.world_y_limit = 3
        self.edge_buffer = 0.5 # Distance from edge to avoid spawning in
        self.max_steps = 500

        # State update parameters
        self.tau = 0.02  # seconds between state updates
        self.kinematics_integrator = 'euler'

        # Meters to pixels conversion for render
        self.scale = 150

        # Set constraints on action and observation spaces
        # action_upper_lim = np.array([self.max_linear_vel, self.max_angular_vel],
        #                             dtype=np.float32)
        # action_lower_lim = np.array([0, -self.max_angular_vel],
        #                             dtype=np.float32)
        obs_lim = np.array([self.world_x_limit,
                            self.world_y_limit,
                            np.pi,
                            self.world_x_limit,
                            self.world_y_limit],
                           dtype=np.float32)
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(-obs_lim, obs_lim,
                                            dtype=np.float32)

        self.seed()
        self.viewer = None
        self.state = None
        self.steps_beyond_done = None
        self.prev_dist = None
        self.prev_angle_offset = None
        self.num_steps = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _compute_angle_offset(self, r_x, r_y, r_theta, g_x, g_y):
        # Compute how close to the desired heading we're currently at
        # 0 means the robot is headed directly towards the goal, -pi or pi
        # means the robot is headed directly away from the goal

        angle_to_goal = math.atan2(g_y-r_y, g_x-r_x) # In world coords

        # Compute angle difference
        # positive offset = right turn needed
        # negative offset = left turn needed
        angle_offset = r_theta - angle_to_goal

        # Keep angle between -pi and pi
        angle_offset = self._restrict_angle_range(angle_offset)

        # Only record magnitude
        return abs(angle_offset)

    def _compute_distance(self, r_x, r_y, g_x, g_y):
        dist = np.linalg.norm([r_x-g_x, r_y-g_y]) # Euclidian distance
        return dist

    def _restrict_angle_range(self, theta):
        # Keep theta within -pi to pi range
        theta = theta % (2*np.pi)
        if theta > np.pi:
            theta -= 2*np.pi
        return theta

    def step(self, action):
        self.num_steps += 1
        # Update state ---------------------------------------------------------
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg

        if action == 0: # Drive straight
            lin_vel = self.max_linear_vel
            ang_vel = 0
        elif action == 1: # Turn left
            lin_vel = 0
            ang_vel = self.max_angular_vel
        elif action == 2: # Turn right
            lin_vel = 0
            ang_vel = -self.max_angular_vel
        else:
            # Prior assert should ensure we never reach this case
            print("Invalid action")

        r_x, r_y, r_theta, g_x, g_y = self.state

        r_theta += self.tau * ang_vel
        # Keep theta within -pi to pi range
        r_theta = self._restrict_angle_range(r_theta)

        r_x += self.tau * lin_vel * np.cos(r_theta)
        r_y += self.tau * lin_vel * np.sin(r_theta)

        self.state = (r_x, r_y, r_theta, g_x, g_y)

        # Update reward --------------------------------------------------------
        # Check if robot is within goal
        curr_dist = self._compute_distance(r_x, r_y, g_x, g_y)
        within_goal = bool(curr_dist < (self.goal_diameter/2.0 - self.robot_diameter/2.0))

        # Check if we left the field
        outside_limit = bool(
            r_x < 0
            or r_x > self.world_x_limit
            or r_y < 0
            or r_y > self.world_y_limit)

        # Check if we've gone over our time limit
        over_time = bool(self.num_steps > self.max_steps)

        # Compute angle offset
        curr_angle_offset = self._compute_angle_offset(r_x, r_y, r_theta, g_x, g_y)

        # Compute rewards
        if not within_goal and not outside_limit and not over_time:
            # We're not done yet - neither terminating case has been reached
            done = False
            dist_delta = self.prev_dist - curr_dist
            angle_delta = self.prev_angle_offset - curr_angle_offset

            if angle_delta > 0.5:
                print("Super big delta detected!")

            reward = (dist_delta * self.distance_reward) + (angle_delta * self.angle_reward) + self.time_reward

            self.prev_dist = curr_dist
            self.prev_angle_offset = curr_angle_offset

        elif self.steps_beyond_done is None:
            # We just reached a terminating case
            done = True
            if outside_limit:
                reward = self.exit_reward
                self.steps_beyond_done = 0

            elif within_goal:
                reward = self.goal_reward
                self.steps_beyond_done = 0

            elif over_time:
                reward = self.exit_reward
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
        self.state = [self.np_random.uniform(low=self.edge_buffer,
                                             high=self.world_x_limit-self.edge_buffer),
                      self.np_random.uniform(low=self.edge_buffer,
                                             high=self.world_y_limit-self.edge_buffer),
                      self.np_random.uniform(low=-np.pi,
                                             high=np.pi),
                      self.np_random.uniform(low=self.edge_buffer,
                                             high=self.world_x_limit-self.edge_buffer),
                      self.np_random.uniform(low=self.edge_buffer,
                                             high=self.world_y_limit-self.edge_buffer)]
        self.steps_beyond_done = None
        self.num_steps = 0
        r_x, r_y, r_theta, g_x, g_y = self.state
        self.prev_angle_offset = self._compute_angle_offset(r_x, r_y, r_theta, g_x, g_y)
        self.prev_dist = self._compute_distance(r_x, r_y, g_x, g_y)
        return np.array(self.state)

    def render(self, mode='human'):
        screen_width = int(self.scale * self.world_x_limit)
        screen_height = int(self.scale * self.world_y_limit)

        world_width = self.world_x_limit
        world_height = self.world_y_limit

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)

            goal = rendering.make_circle(self.scale * self.goal_diameter/2.0)
            self.goaltrans = rendering.Transform()
            goal.add_attr(self.goaltrans)
            goal.set_color(0.1, 0.5, 0.1)
            self.viewer.add_geom(goal)

            robot = rendering.make_circle(self.scale * self.robot_diameter/2.0)
            self.robottrans = rendering.Transform()
            robot.add_attr(self.robottrans)
            robot.set_color(0.0, 0.1, 0.5)
            self.viewer.add_geom(robot)

            direction = rendering.make_polyline([(0, 0),
                (self.scale * self.robot_diameter/2.0, 0)])
            self.directiontrans = rendering.Transform()
            direction.add_attr(self.directiontrans)
            direction.set_color(0.8, 0.8, 0.8)
            self.viewer.add_geom(direction)

        if self.state is None:
            return None

        x = self.state

        robotx = x[0] * self.scale
        roboty = x[1] * self.scale
        goalx = x[3] * self.scale
        goaly = x[4] * self.scale
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
    sample_action = 0#nav_env.action_space.sample()

    while True:
        nav_env.render()
        time.sleep(0.2)
        state, reward, done, _ = nav_env.step(sample_action)
        print(reward)
        if done:
            nav_env.render()
            while True:
                pass
