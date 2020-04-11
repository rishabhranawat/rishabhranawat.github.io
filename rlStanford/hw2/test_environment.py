class TestEnvironment:

	def __init__(self):
		self.states = [0, 1, 2, 3]
		self.actions = [0, 1, 2, 3, 4]
		
		self.state = 0

	def step(self, action):
		current_state = self.state

		next_state = None
		reward = 0.0

		if(action >= 0 and action <= 3):
			next_state = action
		elif(action == 4):
			next_state = 4

		if(next_state == 0):
			reward = 0.1
		elif(next_state == 1):
			reward = -0.2
		elif(next_state == 2):
			reward = 0
		elif(next_state == 3):
			reward = -0.1

		if(current_state == 2):
			reward = reward*(-10)

		self.state = next_state
		return next_state, reward

	def setState(self, state):
		self.state = state

	def getState(self):
		return self.state

	def getActions(self):
		return self.actions

	def getStates(self):
		return self.states


env = TestEnvironment()
actions = env.getActions()
states = env.getStates()

maxReward = 0.0
transition = []
rewards = []

def takeNextStep(env, episode, reward):
	if(len(episode) == 5):
		rewards.append(reward)
		print(max(rewards))
		return rewards
	
	for possibleAction in actions:
		
		updatedEnv = TestEnvironment()
		updatedEnv.setState(env.getState())

		next_state, update_reward = updatedEnv.step(possibleAction)
		updated_episode = [each for each in episode]
		updated_episode.append({next_state, reward+update_reward})
		takeNextStep(updatedEnv, updated_episode, reward+update_reward)



print(takeNextStep(env, [], 0.0))



