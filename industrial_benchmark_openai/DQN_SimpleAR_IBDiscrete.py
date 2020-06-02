"""
DQN_Simple_IBDiscrete.py
by Anurag Kumar
https://github.com/code4funn/industrialbenchmark
You may use, but please credit the source.
"""

import numpy as np
import keras.backend.tensorflow_backend as backend
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
import tensorflow as tf
from collections import deque
import time
import random
from tqdm import tqdm
import os
from PIL import Image
import cv2

# import the openAI IB wrapper
from OpenAI_IB import OpenAI_IB

DISCOUNT = 0.99
REPLAY_MEMORY_SIZE = 50_000  # How many last steps to keep for model training
MIN_REPLAY_MEMORY_SIZE = 1_000  # Minimum number of steps in a memory to start training
MINIBATCH_SIZE = 64 * 4 # How many steps (samples) to use for training
UPDATE_TARGET_EVERY = 5  # Terminal states (end of episodes)
MODEL_NAME = 'IB_disc_AR_k3_b4_100_'
MIN_REWARD = 10_000  # For model save
MEMORY_FRACTION = 0.20
K_SEQ_STATES = 3  # A sequence of k states are sampled randomly from stack

# Environment settings
EPISODES = 1_000

# Exploration settings
epsilon = 1  # not a constant, going to be decayed
EPSILON_DECAY = 0.99 # 0.9975
MIN_EPSILON = 0.001

#  Stats settings
AGGREGATE_STATS_EVERY = 1 #50  # episodes
SHOW_PREVIEW = False




# create an environment object
env = OpenAI_IB(setpoint=100, reward_type='classic', action_type='discrete', stationary_p=True)
nb_actions = env.action_space.n
env.seed(123)

# # For stats
ep_rewards = []

# For more repetitive results
random.seed(123)
np.random.seed(123)
tf.set_random_seed(123)

# Memory fraction, used mostly when trai8ning multiple agents
#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=MEMORY_FRACTION)
#backend.set_session(tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)))

# Create models folder
if not os.path.isdir('models'):
    os.makedirs('models')


# Own Tensorboard class
class ModifiedTensorBoard(TensorBoard):

    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.FileWriter(self.log_dir)

    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        pass

    # Overrided, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Overrided
    # We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    # Overrided, so won't close writer
    def on_train_end(self, _):
        pass

    # Custom method for saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        self._write_logs(stats, self.step)

    def _write_logs(self, logs, index):
        for name, value in logs.items():
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value
            summary_value.tag = name
            # self.writer.add_summary(tf.summary.scalar(name, value),)
            self.writer.add_summary(summary, index)
            # self.step += 1
        self.writer.flush()

    def end(self):
        self.writer.close()

# Agent class
class DQNAgent:
    def __init__(self):

        # Main model  # gets trained every step .fit
        self.model = self.create_model()

        # Target network  # this is what we .predict against every step
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        # An array with last n steps for training
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)  # defined at line 18

        # Custom tensorboard object
        self.tensorboard = ModifiedTensorBoard(log_dir="logs-test/{}-{}".format(MODEL_NAME, int(time.time())))  #
        # @line 22

        # Used to count when to update target network with main network's weights
        self.target_update_counter = 0

    def create_model(self):
        model = Sequential()

        model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
        model.add(Dense(124))
        model.add(Activation('relu'))
        model.add(Dense(56))
        model.add(Activation('relu'))
        model.add(Dense(nb_actions))
        model.add(Activation('linear'))

        model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=['accuracy'])
        return model

    # Adds step's data to a memory replay array
    # (observation space, action, reward, new observation space, done)
    # (array([50., 43., 80., 23.00613734,
    #         11.06587714, 177.39995786, 210.59758929]), 8, -1.9058980607934628, array([50., 42., 90., 28.75767168,
    #                                                                                   5.66814018, 173.58538553,
    #                                                                                   190.58980608]), False)
    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    # Trains main network every step during episode
    def train(self, terminal_state, step):

        # Start training only if certain number of samples is already saved
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        # Get a sequence pf idx of random samples from memory replay table
        miniIdx = random.sample(range(0, len(self.replay_memory) - K_SEQ_STATES), MINIBATCH_SIZE // K_SEQ_STATES + 1)

        # Get a minibatch of random samples from memory replay table
        # minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)
        minibatch = []
        for idx in miniIdx:
            minibatch.extend([self.replay_memory[idx + i] for i in range(K_SEQ_STATES)])

        # Trim minibatch to the size of MINIBATCH
        minibatch = minibatch[:MINIBATCH_SIZE]

        # Get current states from minibatch, then query NN model for Q values
        current_states = np.array([transition[0] for transition in minibatch])
        current_qs_list = self.model.predict(current_states.reshape(-1, 1, 7))

        # Get future states from minibatch, then query NN model for Q values
        # When using target network, query it, otherwise main network should be queried
        new_current_states = np.array([transition[3] for transition in minibatch])
        future_qs_list = self.target_model.predict(new_current_states.reshape(-1, 1, 7))

        X = []  # features
        y = []  # labels/ target values

        # Now we need to enumerate our batches
        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):

            # If not a terminal state, get new q from future states, otherwise set it to 0
            # almost like with Q Learning, but we use just part of equation here
            if not done:
                max_future_q = np.max(future_qs_list[index])
                # print('a ', index, future_qs_list[index])
                # print('b ', np.max(future_qs_list[index]))
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward

            # Update Q value for given state
            current_qs = current_qs_list[index]
            current_qs[action] = new_q
            # print('c ', action)
            # print('d ', current_state)
            # print('e ', current_qs)

            # And append to our training data
            X.append(current_state)
            y.append(current_qs)

        # Fit on all samples as one batch, log only on terminal state
        self.model.fit(np.array(X).reshape(MINIBATCH_SIZE,1,7), np.array(y), batch_size=MINIBATCH_SIZE, verbose=0,
                       shuffle=False,
                       callbacks=[
            self.tensorboard] if terminal_state else None)

        # Update target network counter every episode
        if terminal_state:
            self.target_update_counter += 1

        # If counter reaches set value, update target network with weights of main network
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

    # Queries main network for Q values given current observation space (environment state)
    def get_qs(self, state):
        return self.model.predict(np.array(state).reshape(-1, 1, *state.shape))[0]


# create an agent object
agent = DQNAgent()

# Iterate over episodes
for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):

    # Update tensorboard step every episode
    agent.tensorboard.step = episode

    # Restarting episode - reset episode reward and step number
    episode_reward = 0
    step = 1

    # Reset environment and get initial state
    current_state = env.reset()

    # initialize action
    # action = np.array([0., 0., 0.])

    # Reset flag and start iterating until episode ends
    done = False
    while not done:

        # This part stays mostly the same, the change is to query a model for Q values
        if np.random.random() > epsilon:
            # Get action from Q table
            action = np.argmax(agent.get_qs(current_state))
        else:
            # Get random action
            action = np.random.randint(0, nb_actions)
            # action += 0.1 * (2 * np.random.rand(3) - 1)
            # action = np.clip(action, -1, 1)

        new_state, reward, done, _ = env.step(action) # returns new_state, reward, done, info

        # Transform new continous state to new discrete state and count reward
        episode_reward += reward

        if SHOW_PREVIEW and not episode % AGGREGATE_STATS_EVERY:
            env.render()

        # Every step we update replay memory and train main network
        agent.update_replay_memory((current_state, action, reward, new_state, done))
        agent.train(done, step)

        current_state = new_state
        step += 1

    # Append episode reward to a list and log stats (every given number of episodes)
    ep_rewards.append(episode_reward)
    if not episode % AGGREGATE_STATS_EVERY or episode == 1:
        average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:])/len(ep_rewards[-AGGREGATE_STATS_EVERY:])
        min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
        max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
        agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward, epsilon=epsilon)

        # Save model, but only when min reward is greater or equal a set value
        if min_reward >= MIN_REWARD:
            agent.model.save(f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')

    # Decay epsilon
    if epsilon > MIN_EPSILON:
        epsilon *= EPSILON_DECAY
        epsilon = max(MIN_EPSILON, epsilon)