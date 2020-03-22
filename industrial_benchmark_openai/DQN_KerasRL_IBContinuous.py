'''
The MIT License (MIT)

Copyright 2017 Technical University of Berlin

Authors: Ludwig Winkler

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''
# import numpy as np
#
# from keras.models import Sequential
# from keras.layers import Dense, Activation, Flatten
# from keras.optimizers import Adam
#
# from rl.agents.dqn import DQNAgent
# from rl.policy import BoltzmannQPolicy
# from rl.memory import SequentialMemory
#
# from OpenAI_IB import OpenAI_IB
#
# # Get the environment and extract the number of actions.
# env = OpenAI_IB(setpoint=50, reward_type='classic', action_type='continuous')
# np.random.seed(123)
# env.seed(123)
# nb_actions = env.action_space.shape[0]
# print (nb_actions)
#
#
# print ('step 1')
# obs, reward, done, info = env.step(np.zeros(3))
# print ('obs:')
# print (obs)
# print ('reward: ', reward)
# print ('done: ', done)
# print ('info')
# print (info)
# print (env.action_space)
# print (env.observation_space.shape)
#
#
# # Next, we build a very simple model.
# model = Sequential()
# model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
# model.add(Dense(124))
# model.add(Activation('relu'))
# model.add(Dense(56))
# model.add(Activation('relu'))
# model.add(Dense(nb_actions))
# model.add(Activation('tanh'))
# print(model.summary())
#
# # Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# # even the metrics!
# memory = SequentialMemory(limit=10, window_length=1)
# policy = BoltzmannQPolicy()
# dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=20,
#                target_model_update=1e-2, policy=policy)
# dqn.compile(Adam(lr=1e-3), metrics=['mae'])
#
# # Okay, now it's time to learn something! We visualize the training here for show, but this
# # slows down training quite a lot. You can always safely abort the training prematurely using
# # Ctrl + C.
# dqn.fit(env, nb_steps=5000, visualize=False, verbose=1)


########################################################################################################################

import numpy as np
import gym

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, Concatenate
from keras.optimizers import Adam

from rl.agents import NAFAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess
from rl.core import Processor

from OpenAI_IB import OpenAI_IB

# Get the environment and extract the number of actions.
# env = gym.make(ENV_NAME)
env = OpenAI_IB(setpoint=50, reward_type='classic', action_type='continuous')
np.random.seed(123)
env.seed(123)
assert len(env.action_space.shape) == 1
nb_actions = env.action_space.shape[0]

# Build all necessary models: V, mu, and L networks.
V_model = Sequential()
V_model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
V_model.add(Dense(16))
V_model.add(Activation('relu'))
V_model.add(Dense(16))
V_model.add(Activation('relu'))
V_model.add(Dense(16))
V_model.add(Activation('relu'))
V_model.add(Dense(1))
V_model.add(Activation('linear'))
print(V_model.summary())

mu_model = Sequential()
mu_model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
mu_model.add(Dense(16))
mu_model.add(Activation('relu'))
mu_model.add(Dense(16))
mu_model.add(Activation('relu'))
mu_model.add(Dense(16))
mu_model.add(Activation('relu'))
mu_model.add(Dense(nb_actions))
mu_model.add(Activation('linear'))
print(mu_model.summary())

action_input = Input(shape=(nb_actions,), name='action_input')
observation_input = Input(shape=(1,) + env.observation_space.shape, name='observation_input')
x = Concatenate()([action_input, Flatten()(observation_input)])
x = Dense(32)(x)
x = Activation('relu')(x)
x = Dense(32)(x)
x = Activation('relu')(x)
x = Dense(32)(x)
x = Activation('relu')(x)
x = Dense(((nb_actions * nb_actions + nb_actions) // 2))(x)
x = Activation('linear')(x)
L_model = Model(inputs=[action_input, observation_input], outputs=x)
print(L_model.summary())

# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
# processor = PendulumProcessor()
memory = SequentialMemory(limit=100000, window_length=1)
random_process = OrnsteinUhlenbeckProcess(theta=.15, mu=0., sigma=.3, size=nb_actions)
agent = NAFAgent(nb_actions=nb_actions, V_model=V_model, L_model=L_model, mu_model=mu_model,
                 memory=memory, nb_steps_warmup=100, random_process=random_process,
                 gamma=.99, target_model_update=1e-3)
agent.compile(Adam(lr=.001, clipnorm=1.), metrics=['mae'])

# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using
# Ctrl + C.
agent.fit(env, nb_steps=10000, visualize=False, verbose=1, nb_max_episode_steps=200)

# Finally, evaluate our algorithm for 5 episodes.
agent.test(env, nb_episodes=10, visualize=False, nb_max_episode_steps=200)