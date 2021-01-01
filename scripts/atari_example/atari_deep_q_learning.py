"""
https://github.com/simoninithomas/Deep_reinforcement_learning_Course/blob/master/Deep%20Q%20Learning/Space%20Invaders/DQN%20Atari%20Space%20Invaders.ipynb
Dependencies:
- tensorflow
- gym-retro
- scikit-image


https://github.com/openai/retro/issues/53
    !wget http://www.atarimania.com/roms/Roms.rar
    Make sure that Roms.rar appeared in files
    !unrar x /content/Roms.rar
    There are must be 2 zip archives: ROMS.zip and HC ROMS.zip
    !unzip /content/ROMS.zip
    ROMS folder must appear
    !pip install gym-retro
    Install retro package
    !python3 -m retro.import ROMS/
    Import environments
    env = retro.make(game="SpaceInvaders-Atari2600")
    Create environment!
"""

import tensorflow as tf      # Deep Learning library
import numpy as np           # Handle matrices
import retro                 # Retro Environment


from skimage import transform # Help us to preprocess the frames
from skimage.color import rgb2gray # Help us to gray our frames

import matplotlib.pyplot as plt # Display graphs

from collections import deque# Ordered collection with ends

import random

import warnings # This ignore all the warning messages that are normally printed during the training because of skiimage
warnings.filterwarnings('ignore')

# Custom imports
from dq_network import DQNetwork
from helper_functions import Memory

ENV_NAME = "SpaceInvaders-Atari2600"

class AtariDeepQAgent():
    def __init__(self):
        # Create our environment
        self.env = retro.make(game='SpaceInvaders-Atari2600')
        print("The size of our frame is: ", self.env.observation_space)
        print("The action size is : ", self.env.action_space.n)

        # Hyperparameters ------------------------------------------------------
        ### MODEL HYPERPARAMETERS
        state_size = [110, 84, 4]      # Our input is a stack of 4 frames hence 110x84x4 (Width, height, channels)
        action_size = self.env.action_space.n # 8 possible actions
        learning_rate =  0.00025      # Alpha (aka learning rate)

        ### TRAINING HYPERPARAMETERS
        total_episodes = 50            # Total episodes for training
        max_steps = 50000              # Max possible steps in an episode
        batch_size = 64                # Batch size

        # Exploration parameters for epsilon greedy strategy
        explore_start = 1.0            # exploration probability at start
        explore_stop = 0.01            # minimum exploration probability
        decay_rate = 0.00001           # exponential decay rate for exploration prob

        # Q learning hyperparameters
        gamma = 0.9                    # Discounting rate

        ### MEMORY HYPERPARAMETERS
        pretrain_length = batch_size   # Number of experiences stored in the Memory when initialized for the first time
        memory_size = 1000000          # Number of experiences the Memory can keep

        ### PREPROCESSING HYPERPARAMETERS
        stack_size = 4                 # Number of frames stacked

        ### MODIFY THIS TO FALSE IF YOU JUST WANT TO SEE THE TRAINED AGENT
        training = False

        ## TURN THIS TO TRUE IF YOU WANT TO RENDER THE ENVIRONMENT
        episode_render = False
        # ----------------------------------------------------------------------

        # Here we create an hot encoded version of our actions
        # possible_actions = [[1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0]...]
        # shape = (8,8)
        # Q: what are the 8 actions?
        self.possible_actions = np.array(np.identity(self.env.action_space.n,dtype=int).tolist())

        self.stack_size = 4 # We stack 4 frames
        # Initialize deque with zero-images one array for each image
        self.stacked_frames  =  deque([np.zeros((110,84), dtype=np.int) for i in range(self.stack_size)], maxlen=4)

        # Instantiate deep Q network
        self.network = DQNetwork(state_size, action_size, learning_rate)

    def preprocess_frame(self, frame):
        """
        preprocess_frame: Take a frame,grayscale it, resize it
            __________________
            |                 |
            |                 |
            |                 |
            |                 |
            |_________________|

            to
            _____________
            |            |
            |            |
            |            |
            |____________|
        Normalize it.

        returns preprocessed_frame
        """
        # Greyscale frame
        gray = rgb2gray(frame)

        # Crop the screen (remove the part below the player)
        # [Up: Down, Left: right]
        cropped_frame = gray[8:-12,4:-12]

        # Normalize Pixel Values
        normalized_frame = cropped_frame/255.0

        # Resize
        # Thanks to Mikołaj Walkowiak
        preprocessed_frame = transform.resize(normalized_frame, [110,84])

        return preprocessed_frame # 110x84x1 frame

    def stack_frames(self, stacked_frames, state, is_new_episode):
        # Preprocess frame
        frame = self.preprocess_frame(state)

        if is_new_episode:
            # Clear our stacked_frames
            stacked_frames = deque([np.zeros((110,84), dtype=np.int) for i in range(self.stack_size)], maxlen=4)

            # Because we're in a new episode, copy the same frame 4x
            stacked_frames.append(frame)
            stacked_frames.append(frame)
            stacked_frames.append(frame)
            stacked_frames.append(frame)

            # Stack the frames
            stacked_state = np.stack(stacked_frames, axis=2)

        else:
            # Append frame to deque, automatically removes the oldest frame
            stacked_frames.append(frame)

            # Build the stacked state (first dimension specifies different frames)
            stacked_state = np.stack(stacked_frames, axis=2)

        return stacked_state, stacked_frames

    def instantiate_memory(self):
        # Instantiate memory
        memory = Memory(max_size = memory_size)
        for i in range(pretrain_length):
            # If it's the first step
            if i == 0:
                state = env.reset()

                state, stacked_frames = stack_frames(stacked_frames, state, True)

            # Get the next_state, the rewards, done by taking a random action
            choice = random.randint(1,len(possible_actions))-1
            action = possible_actions[choice]
            next_state, reward, done, _ = env.step(action)

            #env.render()

            # Stack the frames
            next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)


            # If the episode is finished (we're dead 3x)
            if done:
                # We finished the episode
                next_state = np.zeros(state.shape)

                # Add experience to memory
                memory.add((state, action, reward, next_state, done))

                # Start a new episode
                state = env.reset()

                # Stack the frames
                state, stacked_frames = stack_frames(stacked_frames, state, True)

            else:
                # Add experience to memory
                memory.add((state, action, reward, next_state, done))

                # Our new state is now the next_state
                state = next_state

        def instantiate_tensorboard(self):
            # Setup TensorBoard Writer
            writer = tf.summary.FileWriter("/tensorboard/dqn/1")

            ## Losses
            tf.summary.scalar("Loss", DQNetwork.loss)

            write_op = tf.summary.merge_all()

        def predict_action(explore_start, explore_stop, decay_rate, decay_step, state, actions):
            ## EPSILON GREEDY STRATEGY
            # Choose action a from state s using epsilon greedy.
            ## First we randomize a number
            exp_exp_tradeoff = np.random.rand()

            # Here we'll use an improved version of our epsilon greedy strategy used in Q-learning notebook
            explore_probability = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * decay_step)

            if (explore_probability > exp_exp_tradeoff):
                # Make a random action (exploration)
                choice = random.randint(1,len(possible_actions))-1
                action = possible_actions[choice]

            else:
                # Get action from Q-network (exploitation)
                # Estimate the Qs values state
                Qs = sess.run(DQNetwork.output, feed_dict = {DQNetwork.inputs_: state.reshape((1, *state.shape))})

                # Take the biggest Q value (= the best action)
                choice = np.argmax(Qs)
                action = possible_actions[choice]


            return action, explore_probability

        def train(self):
            # Saver will help us to save our model
            saver = tf.train.Saver()

            with tf.Session() as sess:
                # Initialize the variables
                sess.run(tf.global_variables_initializer())

                # Initialize the decay rate (that will use to reduce epsilon)
                decay_step = 0

                for episode in range(total_episodes):
                    # Set step to 0
                    step = 0

                    # Initialize the rewards of the episode
                    episode_rewards = []

                    # Make a new episode and observe the first state
                    state = env.reset()

                    # Remember that stack frame function also call our preprocess function.
                    state, stacked_frames = stack_frames(stacked_frames, state, True)

                    while step < max_steps:
                        step += 1

                        #Increase decay_step
                        decay_step +=1

                        # Predict the action to take and take it
                        action, explore_probability = predict_action(explore_start, explore_stop, decay_rate, decay_step, state, possible_actions)

                        #Perform the action and get the next_state, reward, and done information
                        next_state, reward, done, _ = env.step(action)

                        if episode_render:
                            env.render()

                        # Add the reward to total reward
                        episode_rewards.append(reward)

                        # If the game is finished
                        if done:
                            # The episode ends so no next state
                            next_state = np.zeros((110,84), dtype=np.int)

                            next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)

                            # Set step = max_steps to end the episode
                            step = max_steps

                            # Get the total reward of the episode
                            total_reward = np.sum(episode_rewards)

                            print('Episode: {}'.format(episode),
                                          'Total reward: {}'.format(total_reward),
                                          'Explore P: {:.4f}'.format(explore_probability),
                                        'Training Loss {:.4f}'.format(loss))

                            rewards_list.append((episode, total_reward))

                            # Store transition <st,at,rt+1,st+1> in memory D
                            memory.add((state, action, reward, next_state, done))

                        else:
                            # Stack the frame of the next_state
                            next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)

                            # Add experience to memory
                            memory.add((state, action, reward, next_state, done))

                            # st+1 is now our current state
                            state = next_state


                        ### LEARNING PART
                        # Obtain random mini-batch from memory
                        batch = memory.sample(batch_size)
                        states_mb = np.array([each[0] for each in batch], ndmin=3)
                        actions_mb = np.array([each[1] for each in batch])
                        rewards_mb = np.array([each[2] for each in batch])
                        next_states_mb = np.array([each[3] for each in batch], ndmin=3)
                        dones_mb = np.array([each[4] for each in batch])

                        target_Qs_batch = []

                        # Get Q values for next_state
                        Qs_next_state = sess.run(DQNetwork.output, feed_dict = {DQNetwork.inputs_: next_states_mb})

                        # Set Q_target = r if the episode ends at s+1, otherwise set Q_target = r + gamma*maxQ(s', a')
                        for i in range(0, len(batch)):
                            terminal = dones_mb[i]

                            # If we are in a terminal state, only equals reward
                            if terminal:
                                target_Qs_batch.append(rewards_mb[i])

                            else:
                                target = rewards_mb[i] + gamma * np.max(Qs_next_state[i])
                                target_Qs_batch.append(target)


                        targets_mb = np.array([each for each in target_Qs_batch])

                        loss, _ = sess.run([DQNetwork.loss, DQNetwork.optimizer],
                                                feed_dict={DQNetwork.inputs_: states_mb,
                                                           DQNetwork.target_Q: targets_mb,
                                                           DQNetwork.actions_: actions_mb})

                        # Write TF Summaries
                        summary = sess.run(write_op, feed_dict={DQNetwork.inputs_: states_mb,
                                                               DQNetwork.target_Q: targets_mb,
                                                               DQNetwork.actions_: actions_mb})
                        writer.add_summary(summary, episode)
                        writer.flush()

                    # Save model every 5 episodes
                    if episode % 5 == 0:
                        save_path = saver.save(sess, "./models/model.ckpt")
                        print("Model Saved")

        def test(self):
            with tf.Session() as sess:
                total_test_rewards = []

                # Load the model
                saver.restore(sess, "./models/model.ckpt")

                for episode in range(1):
                    total_rewards = 0

                    state = env.reset()
                    state, stacked_frames = stack_frames(stacked_frames, state, True)

                    print("****************************************************")
                    print("EPISODE ", episode)

                    while True:
                        # Reshape the state
                        state = state.reshape((1, *state_size))
                        # Get action from Q-network
                        # Estimate the Qs values state
                        Qs = sess.run(DQNetwork.output, feed_dict = {DQNetwork.inputs_: state})

                        # Take the biggest Q value (= the best action)
                        choice = np.argmax(Qs)
                        action = possible_actions[choice]

                        #Perform the action and get the next_state, reward, and done information
                        next_state, reward, done, _ = env.step(action)
                        env.render()

                        total_rewards += reward

                        if done:
                            print ("Score", total_rewards)
                            total_test_rewards.append(total_rewards)
                            break


                        next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
                        state = next_state

                env.close()
if __name__=="__main__":
    deep_q = AtariDeepQAgent()

    # # Reset the graph
    # tf.reset_default_graph()
    #
    # deep_q.instantiate_memory()
    # deep_q.instantiate_tensorboard()
    # deep_q.train()
