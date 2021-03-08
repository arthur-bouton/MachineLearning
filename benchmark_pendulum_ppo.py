#!/usr/bin/env python3
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from PPO import PPO
import numpy as np
import sys
from pendulum import Pendulum


from keras.layers import Dense


# Actor network:
def actor( states, a_dim ) :

	x = Dense( 100, activation='relu' )( states )
	x = Dense( 100, activation='relu' )( x )
	mu = Dense( a_dim, activation='tanh' )( x )

	x = Dense( 100, activation='relu' )( states )
	x = Dense( 100, activation='relu' )( x )
	sigma = Dense( a_dim, activation='softplus' )( x )

	return mu, sigma


# Critic network:
def critic( states ) :

	x = Dense( 100, activation='relu' )( states )
	x = Dense( 100, activation='relu' )( x )
	V_value = Dense( 1, activation='linear' )( x )

	return V_value


# Parameters for the training:
ENV = Pendulum
EP_LEN = 100 # Maximum number of steps for a single episode
WORKERS = 4 # Number of parallel workers
EPISODES_PER_BATCH = 4
EP_MAX = 300 # Maximum number of episodes for the training
EVAL_FREQ = 1 # Frequency of the policy evaluation

hyper_params = {}
hyper_params['s_dim'] = 2 # Dimension of the state space
hyper_params['a_dim'] = 1 # Dimension of the action space
hyper_params['actor_def'] = actor # The function defining the actor network
hyper_params['critic_def'] = critic # The function defining the critic network
hyper_params['state_scale'] = [ np.pi, 2*np.pi ] # A scalar or a vector to normalize the state
hyper_params['action_scale'] = None # A scalar or a vector to scale the actions
hyper_params['gamma'] = 0.7 # Discount factor applied to the reward
hyper_params['gae_lambda'] = 0.95 # Generalized advantage estimation parameter (0: only next V estimation, 1: only discounted rewards)
hyper_params['learning_rate'] = 1e-3 # Learning rate
hyper_params['minibatch_size'] = 64 # Size of each minibatch
hyper_params['epochs'] = 200 # Number of iteration at each update
hyper_params['epsilon'] = 0.1 # Surrogate objective clipping parameter
hyper_params['vf_coeff'] = 1 # Critic loss coefficient in the objective function
hyper_params['entropy_coeff'] = 0.01 # Coefficient of the entropy bonus in the objective function

ppo = PPO( **hyper_params )


def trial_generator( worker_id ) :

	training_env = ENV()
	s = training_env.get_obs()

	for t in range( EP_LEN ) :

		# Choose a random action and execute the next step:
		a = ppo.stoch_action( s )
		s_next, r, ep_done, _ = training_env.step( a )
		
		yield s, a, r, ep_done, s_next

		if ep_done : break

		s = s_next


eval_env = ENV()

n_ep = 0

print( 'It Ep LQ R' )

import time
start = time.time()

# Create worker threads running in background:
with ppo.workers( WORKERS, trial_generator, EPISODES_PER_BATCH ) :

	while n_ep < EP_MAX :

		# Gather new data from the workers:
		n_ep, n_samples = ppo.rollout()

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
