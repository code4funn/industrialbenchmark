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
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

from OpenAI_IB import OpenAI_IB

# Get the environment and extract the number of actions.
env = OpenAI_IB(setpoint=50, reward_type='classic', action_type='discrete')
np.random.seed(123)
env.seed(123)
nb_actions = env.action_space.n
print (nb_actions)


print ('step 1')
obs, reward, done, info =  env.step(0)
print ('obs:')
print (obs)
print ('reward: ', reward)
print ('done: ', done)
print ('info')
print (info)
print (env.action_space)
print (env.observation_space.shape)


# Next, we build a very simple model.
model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
model.add(Dense(124))
model.add(Activation('relu'))
model.add(Dense(56))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('linear'))
print(model.summary())

# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memory = SequentialMemory(limit=10, window_length=1)
policy = BoltzmannQPolicy()
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=20,
               target_model_update=1e-2, policy=policy)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])

# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using
# Ctrl + C.
dqn.fit(env, nb_steps=500, visualize=False, verbose=1)



##########################################################################################
# let generate some data from the simulator from the same env and use model.predict(state)
# I'm using the same code from the example.py to generate the data
n_trajectories = 1
T = 1000

# env = IDS(p=100)
obs_names = ['a1', 'a2', 'a3'] + env.IB.observable_keys
data = np.zeros((n_trajectories, T, len(obs_names)))

action = np.array([0., 0., 0.])
for k in range(n_trajectories):
    # env = IDS(p=50)
    for t in range(T):
        # for continuous action space
        action += 0.1 * (2 * np.random.rand(3) - 1)
        action = np.clip(action, -1, 1)

        # for discrete action space
        # action = np.random.randint(-1, 1, 3)
        markovStates = env.IB.step(action)
        data[k, t, 3:] = env.IB.visibleState()
        data[k, t, 0:3] = action

# now we the data. Lets predict the reward/cost for the data collected
test_states = data[:, :, :7]
# y_hat = np.zeros((n_trajectories, T, 3))
y_hat = dqn.model.predict(test_states[0, 0, :].reshape(1, 1, 7))