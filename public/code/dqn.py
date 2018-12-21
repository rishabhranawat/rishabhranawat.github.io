import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


EPSIODES = 100
class BasicDQN:

	def __init__(self, state_size, action_size):
		self.state_size = state_size
		self.action_size = action_size
		self.memory = deque(maxlen=1000)
		self.gamma = 0.95
		self.learning_rate = 0.001

		self.epsilon = 1.0
		self.epsilon_min = 0.001
		self.epsilon_decay = 0.995
		self.model = self.construct()
	
	def construct(self):
		model = Sequential()

		# (1,24) is the size of the output
		model.add(Dense(24, input_dim=self.state_size, activation='relu'))
		model.add(Dense(24, activation='relu'))
		model.add(Dense(self.action_size, activation='linear'))
		model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
		return model

	def remember(self, state, action, reward, next_state, done):
		self.memory.append((state, action, reward, next_state, done))

	def act(self, state):

		# epsilon greedy policy
		if np.random.rand() <= self.epsilon:
			return random.randrange(self.action_size)

		act_values = self.model.predict(state)
		return np.argmax(act_values[0])

	def replay(self, batch_size):
		minibatch = random.sample(self.memory, batch_size)
		for state, action, reward, next_state, done in minibatch:
			if done:
				target = reward				
			else:
				target = (reward + self.gamma * np.amax(self.model.predict(next_state)[0]))

			target_f = self.model.predict(state)
			target_f[0][action] = target
			self.model.fit(state, target_f, epochs=1, verbose=0)
		if self.epsilon > self.epsilon_min:
			self.epsilon *= self.epsilon_decay

if __name__ == "__main__":
	env = gym.make('CartPole-v1')

	state_size = env.observation_space.shape[0]
	print("State Space Size: {}".format(state_size))

	action_size = env.action_space.n
	print("Action Space Size: {}".format(action_size))
	agent = BasicDQN(state_size, action_size)

	done = False
	batch_size = 32

	for e in range(EPSIODES):
		state = env.reset()
		state = np.reshape(state, [1, state_size])
		for time in range(500):
			action = agent.act(state)
			next_state, reward, done, _ = env.step(action)
			reward = reward if not done else -10
			next_state = np.reshape(next_state, [1, state_size])
			agent.remember(state, action, reward, next_state, done)
			state = next_state

			if done: 
				print("epsiode: {}/{}, score:{}, e:{:.2}" .format(e, EPSIODES, time, agent.epsilon))
				break

			if len(agent.memory) > batch_size:
				agent.replay(batch_size)



