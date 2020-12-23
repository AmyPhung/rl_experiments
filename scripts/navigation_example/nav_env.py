"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R

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
        Type: Box(4)
        Num     Observation               Min                Max
        0       Robot X-Position          -sim x-limit       sim x-limit
        1       Robot Y-Position          -sim y-limit       sim y-limit
        2       Goal X-Position           -sim x-limit       sim x-limit
        3       Goal Y-Position           -sim y-limit       sim y-limit

    Actions:
        Type: Box(2)
        Num   Action                      Min           Max
        0     Linear velocity (m/s)       -0.2          0.5
        1     Steering (rad/s)            -3            3

    Reward:
        Reward is -0.1 for every time step, +10000 for reaching the goal, and
        -10000 for leaving the map, +10 * traversed distance to goal (NOTE: need to save distance covered - negative if move away)

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
        # self.gravity = 9.8
        # self.masscart = 1.0
        # self.masspole = 0.1
        # self.total_mass = (self.masspole + self.masscart)
        # self.length = 0.5  # actually half the pole's length
        # self.polemass_length = (self.masspole * self.length)
        # self.force_mag = 10.0
        self.robot_diameter = 0.1

        # Distance at which to fail the episode
        # self.theta_threshold_radians = 12 * 2 * math.pi / 360
        # self.x_threshold = 2.4
        self.world_x_limit = 3
        self.world_y_limit = 2

        # State update parameters
        self.tau = 0.02  # seconds between state updates
        self.kinematics_integrator = 'euler'

        # Meters to pixels conversion for render
        self.scale = 150

        # Angle limit set to 2 * theta_threshold_radians so failing observation
        # is still within bounds.
        # high = np.array([self.x_threshold * 2,
        #                  np.finfo(np.float32).max,
        #                  self.theta_threshold_radians * 2,
        #                  np.finfo(np.float32).max],
        #                 dtype=np.float32)
        #
        # self.action_space = spaces.Discrete(2)
        # self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        #
        # self.seed()
        self.viewer = None
        # self.state = None
        #
        # self.steps_beyond_done = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        # err_msg = "%r (%s) invalid" % (action, type(action))
        # assert self.action_space.contains(action), err_msg
        #
        # x, x_dot, theta, theta_dot = self.state
        # force = self.force_mag if action == 1 else -self.force_mag
        # costheta = math.cos(theta)
        # sintheta = math.sin(theta)
        #
        # # For the interested reader:
        # # https://coneural.org/florian/papers/05_cart_pole.pdf
        # temp = (force + self.polemass_length * theta_dot ** 2 * sintheta) / self.total_mass
        # thetaacc = (self.gravity * sintheta - costheta * temp) / (self.length * (4.0 / 3.0 - self.masspole * costheta ** 2 / self.total_mass))
        # xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        #
        # if self.kinematics_integrator == 'euler':
        #     x = x + self.tau * x_dot
        #     x_dot = x_dot + self.tau * xacc
        #     theta = theta + self.tau * theta_dot
        #     theta_dot = theta_dot + self.tau * thetaacc
        # else:  # semi-implicit euler
        #     x_dot = x_dot + self.tau * xacc
        #     x = x + self.tau * x_dot
        #     theta_dot = theta_dot + self.tau * thetaacc
        #     theta = theta + self.tau * theta_dot
        #
        # self.state = (x, x_dot, theta, theta_dot)
        #
        # done = bool(
        #     x < -self.x_threshold
        #     or x > self.x_threshold
        #     or theta < -self.theta_threshold_radians
        #     or theta > self.theta_threshold_radians
        # )
        #
        # if not done:
        #     reward = 1.0
        # elif self.steps_beyond_done is None:
        #     # Pole just fell!
        #     self.steps_beyond_done = 0
        #     reward = 1.0
        # else:
        #     if self.steps_beyond_done == 0:
        #         logger.warn(
        #             "You are calling 'step()' even though this "
        #             "environment has already returned done = True. You "
        #             "should always call 'reset()' once you receive 'done = "
        #             "True' -- any further steps are undefined behavior."
        #         )
        #     self.steps_beyond_done += 1
        #     reward = 0.0
        #
        # return np.array(self.state), reward, done, {}
        pass

    def reset(self):
        # self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        # self.steps_beyond_done = None
        # return np.array(self.state)
        pass

    def render(self, mode='human'):
        screen_width = int(self.scale * 2*self.world_x_limit)
        screen_height = int(self.scale * 2*self.world_y_limit)

        world_width = self.world_x_limit * 2
        world_height = self.world_y_limit * 2


        x_scale = screen_width/world_width
        y_scale = screen_height/world_height


        # carty = 100  # TOP OF CART
        # polewidth = 10.0
        # polelen = scale * (2 * self.length)
        # cartwidth = 50.0
        # cartheight = 30.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            robot = rendering.make_circle(self.scale * self.robot_diameter)
            self.robottrans = rendering.Transform()
            robot.add_attr(self.robottrans)
            robot.set_color(0.0, 0.1, 0.5)
            self.viewer.add_geom(robot)

            # l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
            # axleoffset = cartheight / 4.0
            # cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            # self.carttrans = rendering.Transform()
            # cart.add_attr(self.carttrans)
            # self.viewer.add_geom(cart)
            # l, r, t, b = -polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2
            # pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            # pole.set_color(.8, .6, .4)
            # self.poletrans = rendering.Transform(translation=(0, axleoffset))
            # pole.add_attr(self.poletrans)
            # pole.add_attr(self.carttrans)
            # self.viewer.add_geom(pole)
            # self.axle = rendering.make_circle(polewidth/2)
            # self.axle.add_attr(self.poletrans)
            # self.axle.add_attr(self.carttrans)
            # self.axle.set_color(.5, .5, .8)
            # self.viewer.add_geom(self.axle)
            # self.track = rendering.Line((0, carty), (screen_width, carty))
            # self.track.set_color(0, 0, 0)
            # self.viewer.add_geom(self.track)
            #
            # self._pole_geom = pole

        # if self.state is None:
        #     return None

        # Edit the pole polygon vertex
        # pole = self._pole_geom
        # l, r, t, b = -polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2
        # pole.v = [(l, b), (l, t), (r, t), (r, b)]
        #
        # x = self.state
        # cartx = x[0] * scale + screen_width / 2.0  # MIDDLE OF CART
        # self.carttrans.set_translation(cartx, carty)
        self.robottrans.set_translation(100,100)
        # self.poletrans.set_rotation(-x[2])

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

if __name__ == "__main__":
    nav_env = NavEnv()
    # nav_env.reset()
    nav_env.render()
    while True:
        pass
