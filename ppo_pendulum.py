#!/usr/bin/env python3
"""
Train a simple pendulum with the Proximal Policy Optimization (PPO) algorithm and multithreaded workers.

Start a new training by calling this script without argument.
When the algorithm has converged, send SIGINT to stop it and save the model by answering "y" when prompted to.
Then run the script with the word "eval" as first argument in order to evaluate the obtained policy.
The training can be took up with a saved model by calling the script with the word "load" as first argument.

Author: Arthur Bouton [arthur.bouton@gadz.org]

Dependency:
keras 2.3.1
"""
from PPO import PPO
import numpy as np
import sys
import os
from pendulum import Pendulum
from looptools import Loop_handler, Monitor


from keras.layers import Dense


# Actor network:
def actor( states, a_dim ) :

	x = Dense( 100, activation='relu' )( states )
	x = Dense( 100, activation='relu' )( x )
	mu = Dense( a_dim, activation='tanh' )( x )

	sigma = Dense( a_dim, activation='softplus' )( x )

	return mu, sigma


# Critic network:
def critic( states ) :

	x = Dense( 100, activation='relu' )( states )
	x = Dense( 100, activation='relu' )( x )
	V_value = Dense( 1, activation='linear' )( x )

	return V_value


# Identifier name for the data:
data_id = 'test1'

# Name of the files where to store the network parameters by default:
script_name = os.path.splitext( os.path.basename( __file__ ) )[0]

session_dir = './training_data/' + script_name + '/' + data_id

# Parameters for the training:
ENV = Pendulum
EP_LEN = 100 # Maximum number of steps for a single episode
WORKERS = 4 # Number of parallel workers
EPISODES_PER_BATCH = 4
EP_MAX = 200 # Maximum number of episodes for the training
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
#hyper_params['summary_dir'] = '/tmp/' + script_name + '/' + data_id # Directory in which to save the summaries

ppo = PPO( **hyper_params )


if len( sys.argv ) == 1 or sys.argv[1] != 'eval' :

	if len( sys.argv ) > 1 and sys.argv[1] == 'load' :
		if len( sys.argv ) > 2 :
			session_dir = sys.argv[2]
		ppo.load( session_dir + '/session' )


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

	reward_graph = Monitor( titles='Average reward per trial', xlabel='trials', keep=False )

	import time
	start = time.time()

	with Loop_handler() as interruption :

		# Create worker threads running in background:
		with ppo.workers( WORKERS, trial_generator, EPISODES_PER_BATCH ) :

			while not interruption() and n_ep < EP_MAX :

				# Gather new data from the workers:
				n_ep, n_samples = ppo.rollout()

				if interruption() :
					break

				# Train the networks:
				L = ppo.train()

				# Evaluate the policy:
				if n_ep % EVAL_FREQ == 0 :
					eval_env.reset( store_data=True )
					stddev_m = 0
					for t in range( EP_LEN ) :
						a, stddev = ppo.best_action( eval_env.get_obs(), return_stddev=True )
						stddev_m += stddev
						_, _, ep_done, _ = eval_env.step( a )
						if ep_done : break
					stddev_m /= EP_LEN
					print( 'It %i | Ep %i | bs %i | Lt %+8.4f | Sd %+5.2f | ' % ( ppo.n_iter, n_ep, n_samples, L, stddev_m ), end='' )
					eval_env.print_eval()
					sys.stdout.flush()
					ppo.reward_summary( eval_env.get_Rt() )
					reward_graph.add_data( n_ep, eval_env.get_Rt() )


	end = time.time()
	print( 'Elapsed time: %.3f' % ( end - start ) )

	answer = input( '\nSave network parameters as ' + session_dir + '? (y) ' )
	if answer.strip() == 'y' :
		os.makedirs( session_dir, exist_ok=True )
		ppo.save( session_dir + '/session' )
		print( 'Parameters saved.' )
	else :
		answer = input( 'Where to store network parameters? (leave empty to discard data) ' )
		if answer.strip() :
			os.makedirs( answer, exist_ok=True )
			ppo.save( answer + '/session' )
			print( 'Parameters saved in %s.' % answer )
		else :
			print( 'Data discarded.' )

else :


	if len( sys.argv ) > 2 :
		session_dir = sys.argv[2]
	ppo.load( session_dir + '/session' )


	test_env = ENV( 180, store_data=True, include_stddev=True )
	for t in range( EP_LEN ) :
		a, stddev = ppo.best_action( test_env.get_obs(), return_stddev=True )
		_, _, ep_done, _ = test_env.step( a, stddev )
		#_, _, ep_done, _ = test_env.step( ppo.stoch_action( test_env.get_obs() ) )
		if ep_done : break
	print( 'Trial result: ', end='' )
	test_env.print_eval()
	test_env.plot3D( ppo.best_action, ppo.get_value, include_stddev=True )
	test_env.plot_trial( plot_stddev=True )
	test_env.show()


ppo.sess.close()
