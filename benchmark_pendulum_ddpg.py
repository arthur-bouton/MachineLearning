#!/usr/bin/env python3
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from DDPG_vanilla import DDPG
#from DDPG_PER import DDPG
import numpy as np
import sys
from pendulum import Pendulum


from keras.layers import Dense, Concatenate


# Actor network:
def actor( states, a_dim ) :

	x = Dense( 100, activation='relu' )( states )
	x = Dense( 100, activation='relu' )( x )
	action = Dense( a_dim, activation='tanh' )( x )

	return action


# Critic network:
def critic( states, actions ) :

	x = Concatenate()( [ states, actions ] )
	x = Dense( 100, activation='relu' )( x )
	x = Dense( 100, activation='relu' )( x )
	Q_value = Dense( 1, activation='linear' )( x )

	return Q_value


# Parameters for the training:
ENV = Pendulum # A class defining the environment
EP_LEN = 100 # Number of steps for one episode
ITER_PER_EP = 200 # Number of training iterations between each episode
EP_MAX = 200 # Maximal number of episodes for the training
EVAL_FREQ = 1 # Frequency of the policy evaluation

hyper_params = {}
hyper_params['s_dim'] = 2 # Dimension of the state space
hyper_params['a_dim'] = 1 # Dimension of the action space
hyper_params['state_scale'] = [ np.pi, 2*np.pi ] # A scalar or a vector to normalize the state
hyper_params['action_scale'] = None # A scalar or a vector to scale the actions
hyper_params['actor_def'] = actor # The function defining the actor network
hyper_params['critic_def'] = critic # The function defining the critic network
hyper_params['gamma'] = 0.7 # Discount factor applied to the reward
hyper_params['tau'] = 1e-3 # Soft target update factor
hyper_params['buffer_size'] = 1e4 # Maximal size of the replay buffer
hyper_params['minibatch_size'] = 64 # Size of each minibatch
hyper_params['actor_lr'] = 1e-3 # Learning rate of the actor network
hyper_params['critic_lr'] = 1e-3 # Learning rate of the critic network
hyper_params['beta_L2'] = 0 # Ridge regularization coefficient
#hyper_params['alpha_sampling'] = 1 # Exponent interpolating between a uniform sampling (0) and a greedy prioritization (1) (DDPG_PER only)
#hyper_params['beta_IS'] = 1 # Exponent of the importance-sampling weights (if 0, no importance sampling) (DDPG_PER only)
hyper_params['seed'] = None # Random seed for the initialization of all random generators
hyper_params['single_thread'] = False # Force the execution on a single core in order to have a deterministic behavior

ddpg = DDPG( **hyper_params )


np.random.seed( hyper_params['seed'] )

training_env = ENV()
eval_env = ENV()

n_ep = 0
L = 0

print( 'It Ep LQ R' )

import time
start = time.time()

while n_ep < EP_MAX :


	# Run a new trial:
	s = training_env.reset()
	exploration = False

	for _ in range( EP_LEN ) :

		# Choose an action:
		if np.random.rand() < 0.5 :
			exploration = not exploration
			if exploration :
				a = np.random.uniform( -1, 1, hyper_params['a_dim'] )
		if not exploration :
			a = ddpg.get_action( s )

		# Do one step:
		s2, r, ep_done, _ = training_env.step( a )

		# Store the experience in the replay buffer:
		ddpg.replay_buffer.append(( s, a, r, ep_done, s2 ))
		
		if ep_done :
			break

		s = s2

	n_ep += 1


	# Train the networks (off-line):
	L = ddpg.train( ITER_PER_EP )


	# Evaluate the policy:
	if L != 0 and n_ep % EVAL_FREQ == 0 :
		s = eval_env.reset( store_data=True )
		for t in range( EP_LEN ) :
			s, _, ep_done, _ = eval_env.step( ddpg.get_action( s ) )
			if ep_done : break
		print( '%i %i %f %f' % ( ddpg.n_iter, n_ep, L, eval_env.get_Rt() ) )
		#sys.stdout.flush()


end = time.time()
print( 'Elapsed time: %.3f' % ( end - start ) )


ddpg.sess.close()
