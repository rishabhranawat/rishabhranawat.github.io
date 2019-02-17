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

	def custom_negative_log_loss(state, y_pred, y_target):
		return -1*keras.backend.log(np.max(y_pred)) * y_target

	def network(self):
		model = Sequential()
		model.add(Dense(24, input_dim=self.state_size, activation='relu'))
		model.add(Dense(24, activation='relu'))
		model.add(Dense(self.action_size, activation='softmax'))
		model.compile(loss=self.custom_negative_log_loss, optimizer=Adam(lr=self.learning_rate))

		return model

	def act(self, state):
		actions = self.network.predict(state)
		return actions

	def update(self, state, target):
		self.network.fit(state, target, verbose=0)
		

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

	def custom_mse(state, y_pred, y_target):
		loss = keras.losses.mean_squared_error(y_pred, y_target)
		return loss

	def network(self):
		model = Sequential()
		model.add(Dense(1, input_dim=self.state_size))
		model.compile(loss=self.custom_mse, optimizer=Adam(lr=self.learning_rate))
		return model

	def act(self, state):
		expected_to_go_reward = self.network.predict(state)
		return expected_to_go_reward

	def update(self, state, target):
		self.network.fit(state, target, verbose=0)

class ActorCritic:

	def __init__(self, state_size, action_size, learning_rate):
		actor = Actor(state_size, action_size, learning_rate)
		critic = Critic(state_size, action_size, learning_rate)

		self.actor = actor
		self.critic = critic

if __name__ == "__main__":
	EPISODES = 100
	BATCH_SAMPLES = 10
	TIME_STEPS = 1000

	# env initializations
	env = gym.make('CartPole-v1')
	state_size = env.observation_space.shape[0]
	action_size = env.action_space.n

	# agent initializations
	ac = ActorCritic(state_size, action_size, 0.01)

	for each_episode in range(EPISODES):
		state = env.reset()
		

		for each_time_step in range(TIME_STEPS):
			state = np.reshape(state, [1, state_size])
			
			original_state = state
			
			advantage = 0.0
			one_step_look = 0.0
			for each_batch_step in range(BATCH_SAMPLES):

				actions = ac.actor.act(state)
				next_state, reward, done, _ = env.step(np.argmax(actions))
				
				next_state = np.reshape(next_state, [1, state_size])

				critic_evaluation_state = ac.critic.act(state)
				critic_evaluation_next_state = ac.critic.act(next_state)

				advantage += reward + 0.9*critic_evaluation_next_state[0][0] - critic_evaluation_state[0][0]
				
				# evaluating the advantage function
				one_step_look += reward + 0.9*critic_evaluation_state[0][0]

				state = next_state

			next_state, reward, done, _ = env.step(np.argmax(actions))

			next_state = np.reshape(next_state, [1, state_size])

			# update the actor and the critic
			ac.critic.update(original_state, np.reshape(one_step_look, (1, 1)))
			ac.actor.update(original_state, np.reshape(advantage, (1, 1)))
			
			if(done):
				print("epsiode: {}/{}, score:{}" .format(each_episode, EPISODES, each_time_step))
				break

			state = next_state









