import random

import gym

import numpy as np
from collections import deque

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

class Actor:

	def __init__(self, state_size, action_size):
		self.state_size = state_size
		self.action_size = action_size
		pass

	def network():

		model = Sequential()

		model.add(Dense(24, input_dim=, activation='relu'))
		model.add(Dense(24, activation='relu'))
		model.add(Dense(self.action_size, activation='linear'))
		model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))

		return model

	def act():
		return

	def update():
		return

	def remember():
		return

class Critic:

	def __init__(self, state_size, action_size):
		pass

	def network():
		return

	def act():
		return

	def update():
		return

	def remember():
		return

class ActorCritic:

	def __init__(self):
		actor = Actor()
		critic = Critic()

	def __init__(self, actor, critic):
		self.actor = actor
		self.critic = critic



if __name__ == "main":
	EPISODES = 1000
	BATCH_SAMPLES = 100
	TIME_STEPS = 1000

	# env initializations
	env = gym.make("Cartpole-v1")
	state_size = env.observation_space.shape[0]
	action_size = env.action_space.n

	# agent initializations
	ac = ActorCritic()

	for each_episode in range(EPISODES):
		state = env.reset()
		state = np.reshape(state, [1, state_size])

		for each_time_step in range(TIME_STEPS):
			original_state = state
			
			# rollout and collect samples
			sample = []
			to_go_reward = 0.0

			for j in range(BATCH_SAMPLES):
				action = ac.actor.act(state)
				next_state, reward, done, _ = env.step(action)
				to_go_reward += reward
				sample.append((state, action, reward, next_state))
				state = next_state

			# update the critic (policy evaluator)
			ac.critic.update(original_state, to_go_reward)

			# evaluating the advantage function
			advantages = []
			for k in range(BATCH_SAMPLES):
				state, action, reward, next_state = sample[k][0], sample[k][1], sample[k][2], sample[k][3]
				advantage = reward + ac.critic.act(next_state) - ac.critic.action_size(state)
				advantages.append(advantage)











