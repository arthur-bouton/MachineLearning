#!/usr/bin/env python3
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from SAC import SAC
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

	mu = layers.Dense( a_dim, activation='linear' )( x )

	x = layers.Dense( 200, activation='relu' )( states )
	x = layers.Dense( 200, activation='relu' )( x )

	sigma = layers.Dense( a_dim, activation='softplus' )( x )

	return keras.Model( states, [ mu, sigma ] )


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
hyper_params['actor_def'] = actor # The function defining the actor model
hyper_params['critic_def'] = critic # The function defining the critic model
hyper_params['gamma'] = 0.9 # Discount factor applied to the reward
#hyper_params['target_entropy'] = -1 # Desired target entropy of the policy
#hyper_params['tau'] = 5e-3 # Soft target update factor
#hyper_params['buffer_size'] = 1e6 # Maximal size of the replay buffer
hyper_params['minibatch_size'] = 64 # Size of each minibatch
#hyper_params['learning_rate'] = 3e-4 # Default learning rate used for all the networks
hyper_params['alpha0'] = 0.1 # Initial value of the entropy temperature
hyper_params['seed'] = None # Random seed for the initialization of all random generators

sac = SAC( **hyper_params )


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

	for _ in range( EP_LEN ) :

		# Choose a random action and execute the next step:
		a = sac.stoch_action( s )
		s2, r, ep_done, _ = training_env.step( a )

		# Store the experience in the replay buffer:
		sac.replay_buffer.append(( s, a, r, ep_done, s2 ))
		
		if ep_done :
			break

		s = s2

	n_ep += 1


	# Train the networks (off-line):
	Q_loss = sac.train( ITER_PER_EP )


	# Evaluate the policy:
	if Q_loss != 0 and n_ep % EVAL_FREQ == 0 :
		eval_env.reset( store_data=True )
		for t in range( EP_LEN ) :
			a, stddev = sac.best_action( eval_env.get_obs(), return_stddev=True )
			_, _, ep_done, _ = eval_env.step( a )
			if ep_done : break
		print( '%i %i %f %f' % ( sac.n_iter, n_ep, Q_loss, eval_env.get_Rt() ) )
		#sys.stdout.flush()


end = time.time()
print( 'Elapsed time: %.3f' % ( end - start ) )
