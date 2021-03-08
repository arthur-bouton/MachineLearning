#!/usr/bin/env python3
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from TD3 import TD3
import numpy as np
import sys
from cartpole import Cartpole


from tensorflow import keras
from tensorflow.keras import layers


# Actor network:
def actor( s_dim, a_dim ) :

	states = keras.Input( shape=s_dim )

	x = layers.Dense( 200, activation='relu' )( states )
	x = layers.Dense( 200, activation='relu' )( x )

	actions = layers.Dense( a_dim, activation='tanh' )( x )

	return keras.Model( states, actions )


# Critic network:
def critic( s_dim, a_dim ) :

	states  = keras.Input( shape=s_dim )
	actions = keras.Input( shape=a_dim )

	x = layers.Concatenate()( [ states, actions ] )
	x = layers.Dense( 200, activation='relu' )( x )
	x = layers.Dense( 200, activation='relu' )( x )
	Q_value = layers.Dense( 1, activation='linear' )( x )

	return keras.Model( [ states, actions ], Q_value )


# Parameters for the training:
ENV = Cartpole # A class defining the environment
EP_LEN = 200 # Number of steps for one episode
ITER_PER_EP = 200 # Number of training iterations between each episode
EP_MAX = 600 # Maximal number of episodes for the training
EVAL_FREQ = 1 # Frequency of the policy evaluation

hyper_params = {}
hyper_params['s_dim'] = 4 # Dimension of the state space
hyper_params['a_dim'] = 1 # Dimension of the action space
hyper_params['state_scale'] = [ 1, 1, np.pi, 2*np.pi ] # A scalar or a vector to normalize the state
hyper_params['action_scale'] = None # A scalar or a vector to scale the actions
hyper_params['actor_def'] = actor # The function defining the actor network
hyper_params['critic_def'] = critic # The function defining the critic network
hyper_params['gamma'] = 0.9 # Discount factor applied to the reward
#hyper_params['tau'] = 5e-3 # Soft target update factor
#hyper_params['policy_update_delay'] = 2 # Number of critic updates for one policy update
#hyper_params['policy_reg_sigma'] = 0.2 # Standard deviation of the target policy regularization noise
#hyper_params['policy_reg_bound'] = 0.5 # Bounds of the target policy regularization noise
#hyper_params['buffer_size'] = 1e4 # Maximal size of the replay buffer
hyper_params['minibatch_size'] = 64 # Size of each minibatch
#hyper_params['learning_rate'] = 1e-3 # Default learning rate used for all the networks
hyper_params['seed'] = None # Random seed for the initialization of all random generators

td3 = TD3( **hyper_params )


np.random.seed( hyper_params['seed'] )

training_env = ENV()
eval_env = ENV()

n_ep = 0
Q_loss = 0

print( 'It Ep LQ R' )

import time
start = time.time()

while n_ep < EP_MAX :


	# Run a new trial:
	s = training_env.reset()
	exploration = False

	for _ in range( EP_LEN ) :

		# Choose an action:
		draw = np.random.rand()
		if not exploration and draw < 0.1 or exploration and draw < 0.5 :
			exploration = not exploration
			if exploration :
				a = np.random.uniform( -1, 1, hyper_params['a_dim'] )
		if not exploration :
			a = td3.get_action( s )

		# Do one step:
		s2, r, ep_done, _ = training_env.step( a )

		# Store the experience in the replay buffer:
		td3.replay_buffer.append(( s, a, r, ep_done, s2 ))
		
		if ep_done :
			break

		s = s2

	n_ep += 1

	# Train the networks (off-line):
	Q_loss = td3.train( ITER_PER_EP )


	# Evaluate the policy:
	if Q_loss != 0 and n_ep % EVAL_FREQ == 0 :
		s = eval_env.reset( store_data=True )
		for t in range( EP_LEN ) :
			s, _, ep_done, _ = eval_env.step( td3.get_action( s ) )
			if ep_done : break
		print( '%i %i %f %f' % ( td3.n_iter, n_ep, Q_loss, eval_env.get_Rt() ) )
		#sys.stdout.flush()


end = time.time()
print( 'Elapsed time: %.3f' % ( end - start ) )
