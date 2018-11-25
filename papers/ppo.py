import tensorflow as tf
import numpy as np
import gym


def learn(env, policy_fn):

	ob_space = env.observation_space
	ac_space = env.action_space

env = 