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
        Type: Discrete()
        Num     Observation               Min                Max
        0       Robot X-Position          -sim x-limit       sim x-limit
        1       Robot Y-Position          -sim y-limit       sim y-limit
        2       Robot direction           -pi                pi
        3       Goal X-Position           -sim x-limit       sim x-limit
        4       Goal Y-Position           -sim y-limit       sim y-limit

    Actions:
        Type: Discrete(3)
        Num   Action
        0     Move straight 1 meter
        1     Turn right (-90 degrees)
        2     Turn left (90 degrees)

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
        # self.robot_diameter = 0.3 # meters
        # self.goal_diameter = 1.0 # meters
        # self.max_linear_vel = 0.4 # meters/sec
        # self.max_angular_vel = 5 # radians/sec

        # Rewards
        self.goal_reward = 0
        self.exit_reward = 0
        self.time_reward = -100
        self.distance_reward = 0

        # Distance at which to fail the episode
        # Note: only use ints - discrete case
        self.world_x_limit = 10
        self.world_y_limit = 5
        self.max_steps = 500

        # State update parameters
        self.tau = 1  # seconds between state updates
        self.kinematics_integrator = 'euler'

        # Meters to pixels conversion for render
        self.scale = 100

        # # Set constraints on action and observation spaces
        # action_lim = np.array([self.max_linear_vel,
        #                        self.max_angular_vel],
        #                       dtype=np.float32)
        # obs_lim = np.array([self.world_x_limit,
        #                     self.world_y_limit,
        #                     np.pi,
        #                     self.world_x_limit,
        #                     self.world_y_limit],
        #                    dtype=np.float32)
        self.action_space = spaces.Discrete(3) # Turn left, turn right, go straight
        self.observation_space = spaces.Discrete(self.world_x_limit * self.world_y_limit * 4) # X, Y, theta

        self.seed()
        self.viewer = None
        self.state = None
        self.steps_beyond_done = None
        self.prev_dist = None
        self.num_steps = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        self.num_steps += 1
        # Update state ---------------------------------------------------------
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg

        if action == 0: # Drive straight
            lin_vel = 1
            ang_vel = 0
        elif action == 1: # Turn right
            lin_vel = 0
            ang_vel = -1
        elif action == 2: # Turn left
            lin_vel = 0
            ang_vel = 1
        else:
            print("Invalid action")

        r_x, r_y, direction, g_x, g_y = self.state

        # Update discrete direction
        direction += ang_vel
        if direction == 4: # Loop direction
            direction = 0
        elif direction == -1:
            direction = 3

        r_theta = direction * np.pi/2

        r_x += lin_vel*np.cos(r_theta)
        r_y += lin_vel*np.sin(r_theta)

        self.state = (r_x, r_y, direction, g_x, g_y)

        # Update reward --------------------------------------------------------
        # Check if robot is within goal
        curr_dist = np.linalg.norm([r_x-g_x, r_y-g_y])
        within_goal = bool(curr_dist < 0.1)

        # Check if we hit the edge of field
        outside_limit = bool(
            r_x < 0
            or r_x >= self.world_x_limit
            or r_y < 0
            or r_y >= self.world_y_limit)

        # Check if we've gone over our time limit
        over_time = bool(self.num_steps > self.max_steps)

        # Compute rewards
        if not within_goal and not outside_limit and not over_time:
            # We're not done yet - neither terminating case has been reached
            done = False
            dist_delta = self.prev_dist - curr_dist
            reward = (dist_delta * self.distance_reward) + self.time_reward

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

        return tuple(np.array(self.state, dtype=int)), reward, done, {}

    def reset(self):
        # State: robot-x, robot-y, robot-theta, goal-x, goal-y
        self.state = [1,#self.np_random.randint(low=1, high=self.world_x_limit-1),
                      1,#self.np_random.randint(low=1, high=self.world_y_limit-1),
                      self.np_random.randint(low=1, high=4), # up/down/left/right
                      8,#,self.np_random.randint(low=1, high=self.world_x_limit),
                      3]#self.np_random.randint(low=1, high=self.world_y_limit)]

        self.steps_beyond_done = None
        self.num_steps = 0
        r_x, r_y, r_theta, g_x, g_y = self.state
        self.prev_dist = np.linalg.norm([r_x-g_x, r_y-g_y])
        return tuple(self.state)

    def render(self, mode='human'):
        screen_width = int(self.scale * self.world_x_limit)
        screen_height = int(self.scale * self.world_y_limit)

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            robot = rendering.make_circle(self.scale/2)
            self.robottrans = rendering.Transform()
            robot.add_attr(self.robottrans)
            robot.set_color(0.0, 0.1, 0.5)
            self.viewer.add_geom(robot)

            direction = rendering.make_polyline([(0, 0),
                (self.scale/2, 0)])
            self.directiontrans = rendering.Transform()
            direction.add_attr(self.directiontrans)
            direction.set_color(0.8, 0.8, 0.8)
            self.viewer.add_geom(direction)

            goal = rendering.make_circle(self.scale/2)
            self.goaltrans = rendering.Transform()
            goal.add_attr(self.goaltrans)
            goal.set_color(0.1, 0.5, 0.1)
            self.viewer.add_geom(goal)

        if self.state is None:
            return None

        x = self.state

        robotx = (x[0]+0.5) * self.scale #0.5 to "center" visual
        roboty = (x[1]+0.5) * self.scale
        goalx = (x[3]+0.5) * self.scale
        goaly = (x[4]+0.5) * self.scale
        self.robottrans.set_translation(robotx, roboty)
        self.directiontrans.set_translation(robotx, roboty)
        self.directiontrans.set_rotation(x[2]*np.pi/2)
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

    # sample_action = nav_env.action_space.sample()
    sample_action = 2
    while True:
        time.sleep(0.4)
        nav_env.render()
        state, reward, done, _ = nav_env.step(sample_action)
        print(reward)
        if done:
            while True:
                pass
