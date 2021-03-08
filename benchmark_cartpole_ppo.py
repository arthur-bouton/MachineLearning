#!/usr/bin/env python3
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from PPO import PPO
import numpy as np
import sys
from cartpole import Cartpole


from keras.layers import Dense


# Actor network:
def actor( states, a_dim ) :

	x = Dense( 200, activation='relu' )( states )
	x = Dense( 200, activation='relu' )( x )
	mu = Dense( a_dim, activation='tanh' )( x )

	x = Dense( 200, activation='relu' )( states )
	x = Dense( 200, activation='relu' )( x )
	sigma = Dense( a_dim, activation='softplus' )( x )

	return mu, sigma


# Critic network:
def critic( states ) :

	x = Dense( 200, activation='relu' )( states )
	x = Dense( 200, activation='relu' )( x )
	V_value = Dense( 1, activation='linear' )( x )

	return V_value


# Parameters for the training:
ENV = Cartpole # A class defining the environment
EP_LEN = 200 # Maximum number of steps for a single episode
EPISODES_PER_BATCH = 4
EP_MAX = 600 # Maximum number of episodes for the training
EVAL_FREQ = 1 # Frequency of the policy evaluation

hyper_params = {}
hyper_params['s_dim'] = 4 # Dimension of the state space
hyper_params['a_dim'] = 1 # Dimension of the action space
hyper_params['actor_def'] = actor # The function defining the actor network
hyper_params['critic_def'] = critic # The function defining the critic network
hyper_params['state_scale'] = [ 1, 1, np.pi, 2*np.pi ] # A scalar or a vector to normalize the state
hyper_params['action_scale'] = None # A scalar or a vector to scale the actions
hyper_params['gamma'] = 0.9 # Discount factor applied to the reward
hyper_params['gae_lambda'] = 0.95 # Generalized advantage estimation parameter (0: only next V estimation, 1: only discounted rewards)
hyper_params['learning_rate'] = 1e-4 # Learning rate
hyper_params['minibatch_size'] = 64 # Size of each minibatch
hyper_params['epochs'] = 400 # Number of iteration at each update
hyper_params['epsilon'] = 0.1 # Surrogate objective clipping parameter
hyper_params['vf_coeff'] = 1 # Critic loss coefficient in the objective function
hyper_params['entropy_coeff'] = 0.01 # Coefficient of the entropy bonus in the objective function

ppo = PPO( **hyper_params )


training_env = ENV()
eval_env = ENV()

n_ep = 0

print( 'It Ep LQ R' )

import time
start = time.time()

while n_ep < EP_MAX :


	# Gather new data from the current policy:
	n_samples = 0

	for ep in range( EPISODES_PER_BATCH ) :

		s = training_env.reset()

		for t in range( EP_LEN ) :

			# Choose a random action and execute the next step:
			a = ppo.stoch_action( s )
			s_next, r, ep_done, _ = training_env.step( a )
			
			ppo.add_transitions(( s, a, r, ep_done, s_next ))
			n_samples += 1

			if ep_done : break

			s = s_next

		ppo.process_episode()
		n_ep += 1


	# Train the networks:
	L = ppo.train()

	# Evaluate the policy:
	if n_ep % EVAL_FREQ == 0 :
		eval_env.reset( store_data=True )
		for t in range( EP_LEN ) :
			a, stddev = ppo.best_action( eval_env.get_obs(), return_stddev=True )
			_, _, ep_done, _ = eval_env.step( a )
			if ep_done : break
		print( '%i %i %f %f' % ( ppo.n_iter, n_ep, L, eval_env.get_Rt() ) )
		#sys.stdout.flush()


end = time.time()
print( 'Elapsed time: %.3f' % ( end - start ) )

ppo.sess.close()
