import random

import gym

import numpy as np
from collections import deque

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

'''
The Actor Network is the policy network i.e., for a given state
it will return a vector of probabilities of taking a particular
action.
'''
class Actor:

	def __init__(self, state_size, action_size, learning_rate):
		self.state_size = state_size
		self.action_size = action_size
		self.learning_rate = learning_rate
		self.network = self.network()

	def custom_negative_log_loss(state, actual_action_prob, target):
		return -1*keras.backend.log(actual_action_prob) * target

	def network(self):
		model = Sequential()

		model.add(Dense(24, input_dim=self.state_size, activation='relu'))
		model.add(Dense(24, activation='relu'))
		model.add(Dense(self.action_size, activation='linear'))
		model.compile(loss=self.custom_negative_log_loss, optimizer=Adam(lr=self.learning_rate))

		return model

	def act(self, state):
		actions = self.network.predict(state)
		action = np.argmax(actions[0])
		return action, actions[0][action]

	def update(self, state, target, action_prob):
		self.network.fit(state, action_prob, target, epochs=1, verbose=0)
		

'''
The critic network is the policy evaluator i.e., it is the function approximator
for the value function. Now, the aim of the value function is to give us the
expected to-go reward given a particular policy.

In this case, the critic network will take in the state as an input and give
the expected reward for being in that state.
'''
class Critic:
	def __init__(self, state_size, action_size, learning_rate):
		self.state_size = state_size
		self.action_size = action_size
		self.learning_rate = learning_rate

		self.network = self.network()

	def network(self):
		model = Sequential()
		model.add(Dense(1, input_dim=self.state_size))
		model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
		return model

	def act(self, state):
		expected_to_go_reward = self.network.predict(state)
		return expected_to_go_reward

	def update(self, value_estimate, target):
		self.network.fit(value_estimate, target, epochs=1, verbose=0)

class ActorCritic:

	def __init__(self, state_size, action_size, learning_rate):
		actor = Actor(state_size, action_size, learning_rate)
		critic = Critic(state_size, action_size, learning_rate)

		self.actor = actor
		self.critic = critic



def get_one_hot_state(state, state_size):
	return keras.backend.one_hot(state, state_size)


if __name__ == "__main__":
	EPISODES = 1000
	BATCH_SAMPLES = 100
	TIME_STEPS = 1000

	# env initializations
	env = gym.make('CartPole-v1')
	state_size = env.observation_space.shape[0]
	action_size = env.action_space.n

	# agent initializations
	ac = ActorCritic(state_size, action_size, 0.95)

	for each_episode in range(EPISODES):
		state = env.reset()
		

		for each_time_step in range(TIME_STEPS):
			state = np.reshape(state, [1, state_size])
			
			action, action_prob = ac.actor.act(state)
			next_state, reward, done, _ = env.step(action)
			
			next_state = np.reshape(state, [1, state_size])

			critic_evaluation_state = ac.critic.act(state)
			critic_evaluation_next_state = ac.critic.act(next_state)

			# update the critic (policy evaluator)
			ac.critic.update(critic_evaluation_state, reward)

			# evaluating the advantage function
			advantage = reward + critic_evaluation_next_state - critic_evaluation_state

			ac.actor.update(state, advantage, action_prob)
			
			if(done):
				print("epsiode: {}/{}, score:{}, e:{:.2}" .format(each_episode, EPSIODES, each_time_step))
				break

			state = next_state









