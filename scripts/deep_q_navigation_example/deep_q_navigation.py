"""
From https://github.com/Huixxi/TensorFlow2.0-for-Deep-Reinforcement-Learning/blob/master/01_dqn.py

Takes 320 seconds to train

A simple version of Deep Q-Network(DQN) including the main tactics mentioned in DeepMind's original paper:
- Experience Replay
- Target Network
To play CartPole-v0.
> Note: DQN can only handle discrete-env which have a discrete action space, like up, down, left, right.
        As for the CartPole-v0 environment, its state(the agent's observation) is a 1-D vector not a 3-D image like
        Atari, so in that simple example, there is no need to use the convolutional layer, just fully-connected layer.
Using:
TensorFlow 2.0
Numpy 1.16.2
Gym 0.12.1
"""

import tensorflow as tf
print(tf.__version__)

import gym
import time
import numpy as np
import tensorflow.keras.layers as kl
import tensorflow.keras.optimizers as ko
from data_saver import DataSaver
import time
from nav_env import NavEnv

np.random.seed(1)
tf.random.set_seed(1)

# Neural Network Model Defined at Here.
class Model(tf.keras.Model):
    def __init__(self, num_actions):
        super().__init__(name='basic_dqn')
        # you can try different kernel initializer
        self.fc1 = kl.Dense(32, activation='relu', kernel_initializer='he_uniform')
        self.fc2 = kl.Dense(32, activation='relu', kernel_initializer='he_uniform')
        self.logits = kl.Dense(num_actions, name='q_values')

    # forward propagation
    def call(self, inputs):
        x = self.fc1(inputs)
        x = self.fc2(x)
        x = self.logits(x)
        return x

    # a* = argmax_a' Q(s, a')
    def action_value(self, obs):
        q_values = self.predict(obs)
        best_action = np.argmax(q_values, axis=-1)
        return best_action[0], q_values[0]

# To test whether the model works
def test_model():
    env = NavEnv()
    print('num_actions: ', env.action_space.n)
    model = Model(env.action_space.n)

    obs = env.reset()
    print('obs_shape: ', obs.shape)

    # tensorflow 2.0: no feed_dict or tf.Session() needed at all
    best_action, q_values = model.action_value(obs[None])
    print('res of test model: ', best_action, q_values)  # 0 [ 0.00896799 -0.02111824]


class DQNAgent:  # Deep Q-Network
    def __init__(self, model, target_model, env, buffer_size=100, learning_rate=.0015, epsilon=.1, epsilon_decay=0.995,
                 min_epsilon=.01, gamma=.95, batch_size=4, target_update_iter=400, train_nums=2000, start_learning=10):#700000
        self.model = model
        self.target_model = target_model
        # print(id(self.model), id(self.target_model))  # to make sure the two models don't update simultaneously
        # gradient clip
        opt = ko.Adam(learning_rate=learning_rate, clipvalue=10.0)  # do gradient clip
        self.model.compile(optimizer=opt, loss='mse')

        # parameters
        self.env = env                              # gym environment
        self.lr = learning_rate                     # learning step
        self.epsilon = epsilon                      # e-greedy when exploring
        self.epsilon_decay = epsilon_decay          # epsilon decay rate
        self.min_epsilon = min_epsilon              # minimum epsilon
        self.gamma = gamma                          # discount rate
        self.batch_size = batch_size                # batch_size
        self.target_update_iter = target_update_iter    # target network update period
        self.train_nums = train_nums                # total training steps
        self.num_in_buffer = 0                      # transition's num in buffer
        self.buffer_size = buffer_size              # replay buffer size
        self.start_learning = start_learning        # step to begin learning(no update before that step)

        # replay buffer params [(s, a, r, ns, done), ...]
        self.obs = np.empty((self.buffer_size,) + self.env.reset().shape)
        self.actions = np.empty((self.buffer_size), dtype=np.int8)
        self.rewards = np.empty((self.buffer_size), dtype=np.float32)
        self.dones = np.empty((self.buffer_size), dtype=np.bool)
        self.next_states = np.empty((self.buffer_size,) + self.env.reset().shape)
        self.next_idx = 0

        self.data_saver = DataSaver(self.env, self.model, 'NavEnv', version=3)
        self.data_saver.save_params_to_file(self.to_string())

    def train(self):
        episode = 0
        last_t = 0

        # initialize the initial observation of the agent
        obs = self.env.reset()
        for t in range(1, self.train_nums):
            best_action, q_values = self.model.action_value(obs[None])  # input the obs to the network model
            action = self.get_action(best_action)   # get the real action
            next_obs, reward, done, info = self.env.step(action)    # take the action in the env to return s', r, done
            self.store_transition(obs, action, reward, next_obs, done)  # store that transition into replay butter
            self.num_in_buffer = min(self.num_in_buffer + 1, self.buffer_size)

            # TODO: modify the logger so that the ending episodes don't just have a bunch of 0's
            # would require removing the tuple and using a list isntead in data_saver.py
            self.data_saver.add_data_point(episode, t - last_t, reward)

            if t > self.start_learning:  # start learning
                losses = self.train_step()

                if t % 1000 == 0:
                    print('losses each 1000 steps: ', losses)

            if t % self.target_update_iter == 0:
                self.update_target_model()

            if done:
                # Record checkpoint every few episodes
                if episode % 100 == 0:
                    self.data_saver.record_checkpoint()

                obs = self.env.reset()
                last_t = t
                episode += 1

            else:
                obs = next_obs

        self.data_saver.plot_and_save_stats()
        self.data_saver.record_checkpoint()

    def train_step(self):
        idxes = self.sample(self.batch_size)
        s_batch = self.obs[idxes]
        a_batch = self.actions[idxes]
        r_batch = self.rewards[idxes]
        ns_batch = self.next_states[idxes]
        done_batch = self.dones[idxes]

        target_q = r_batch + self.gamma * np.amax(self.get_target_value(ns_batch), axis=1) * (1 - done_batch)
        target_f = self.model.predict(s_batch)
        for i, val in enumerate(a_batch):
            target_f[i][val] = target_q[i]

        losses = self.model.train_on_batch(s_batch, target_f)

        return losses

    def evaluation(self, env, render=True):
        env = gym.wrappers.Monitor(env, self.data_saver.save_dir, force=True)

        obs, done, ep_reward = env.reset(), False, 0
        # one episode until done
        while not done:
            action, q_values = self.model.action_value(obs[None])  # Using [None] to extend its dimension (4,) -> (1, 4)
            obs, reward, done, info = env.step(action)
            ep_reward += reward
            if render:  # visually show
                env.render()
            # time.sleep(0.05)
        env.close()
        return ep_reward

    # store transitions into replay butter
    def store_transition(self, obs, action, reward, next_state, done):
        n_idx = self.next_idx % self.buffer_size
        self.obs[n_idx] = obs
        self.actions[n_idx] = action
        self.rewards[n_idx] = reward
        self.next_states[n_idx] = next_state
        self.dones[n_idx] = done
        self.next_idx = (self.next_idx + 1) % self.buffer_size

    # sample n different indexes
    def sample(self, n):
        assert n < self.num_in_buffer
        res = []
        while True:
            num = np.random.randint(0, self.num_in_buffer)
            if num not in res:
                res.append(num)
            if len(res) == n:
                break
        return res

    # e-greedy
    def get_action(self, best_action):
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()
        return best_action

    # assign the current network parameters to target network
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def get_target_value(self, obs):
        return self.target_model.predict(obs)

    def e_decay(self):
        self.epsilon *= self.epsilon_decay

    def to_string(self):
        s = "buffer_size=" + str(self.buffer_size) + ", " + \
            "learning_rate=" + str(self.lr) + ", " + \
            "epsilon=" + str(self.epsilon) + ", " + \
            "epsilon_decay=" + str(self.epsilon_decay) + ", " + \
            "min_epsilon=" + str(self.min_epsilon) + ", " + \
            "gamma=" + str(self.gamma) + ", " + \
            "batch_size=" + str(self.batch_size) + ", " + \
            "target_update_iter=" + str(self.target_update_iter) + ", " + \
            "train_nums=" + str(self.train_nums) + ", " + \
            "start_learning=" + str(self.start_learning)
        return s

if __name__ == '__main__':
    test_model()

    env = NavEnv()
    num_actions = env.action_space.n
    model = Model(num_actions)
    target_model = Model(num_actions)
    agent = DQNAgent(model, target_model,  env)

    # print(agent.to_string())
    # # test before
    # rewards_sum = agent.evaluation(env, render=False)
    # print("Before Training: %d out of 200" % rewards_sum) # 9 out of 200

    t_start = time.time()
    agent.train()
    t_end = time.time()
    print ("Training time: " + str(t_end - t_start))

    # # test after
    # rewards_sum = agent.evaluation(env)
    # print("After Training: %d out of 200" % rewards_sum) # 200 out of 200
    # env.close()
