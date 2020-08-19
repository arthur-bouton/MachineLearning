#!/usr/bin/env python3
""" 
Train a simple pendulum with the Deep Deterministic Policy Gradient (DDPG) algorithm.

Start a new training by calling this script without argument.
When the algorithm has converged, send SIGINT to stop it and save the model and the content of the replay buffer by answering "y" when prompted to.
Then run the script with the word "eval" as first argument in order to evaluate the obtained policy.
The training can be took up with a saved model and replay buffer by calling the script with the word "load" as first argument.

Author: Arthur Bouton [arthur.bouton@gadz.org]

Dependency:
tensorflow 1.13.1
"""
from DDPG_vanilla import DDPG
#from DDPG_PER import DDPG
from pendulum import Pendulum
from looptools import Loop_handler, Monitor
import tensorflow as tf
import numpy as np
import sys
import os


sess = tf.Session()

from keras.layers import Dense
from keras import backend as K
K.set_session( sess )


def actor( states, a_dim ) :

	x = Dense( 100, activation='relu' )( states )
	x = Dense( 100, activation='relu' )( x )
	action = Dense( a_dim, activation='tanh' )( x )

	return action


def critic( states, actions ) :

	x = Dense( 100, activation='relu' )( tf.concat( [ states, actions ], 1 ) )
	x = Dense( 100, activation='relu' )( x )
	Q_value = Dense( 1, activation='linear' )( x )

	return Q_value


# Identifier name for the data:
data_id = 'test1'

script_name = os.path.splitext( os.path.basename( __file__ ) )[0]

# Name of the files where to store the network parameters and replay buffer by default:
session_dir = './training_data/' + script_name + '/' + data_id

# Parameters for the training:
ENV = Pendulum # A class defining the environment
EP_LEN = 100 # Number of steps for one episode
EP_MAX = 2000 # Maximal number of episodes for the training
ITER_PER_EP = 200 # Number of training iterations between each episode
EVAL_FREQ = 1 # Frequency of the policy evaluation

hyper_params = {}
hyper_params['s_dim'] = 2 # Dimension of the state space
hyper_params['a_dim'] = 1 # Dimension of the action space
hyper_params['state_scale'] = [ np.pi, 2*np.pi ] # A scalar or a vector to normalize the state
hyper_params['action_scale'] = None # A scalar or a vector to scale the actions
hyper_params['sess'] = sess # The TensorFlow session to use
hyper_params['actor_def'] = actor # The function defining the actor network
hyper_params['critic_def'] = critic # The function defining the critic network
hyper_params['gamma'] = 0.7 # Discount factor of the reward
hyper_params['tau'] = 1e-3 # Soft target update factor
hyper_params['buffer_size'] = 1e4 # Maximal size of the replay buffer
hyper_params['minibatch_size'] = 64 # Size of each minibatch
hyper_params['actor_lr'] = 1e-3 # Learning rate of the actor network
hyper_params['critic_lr'] = 1e-3 # Learning rate of the critic network
hyper_params['beta_L2'] = 0 # Ridge regularization coefficient
#hyper_params['alpha_sampling'] = 1 # Exponent interpolating between a uniform sampling (0) and a greedy prioritization (1) (DDPG_PER only)
#hyper_params['beta_IS'] = 1 # Exponent of the importance-sampling weights (if 0, no importance sampling) (DDPG_PER only)
hyper_params['summary_dir'] = None # No summaries
#hyper_params['summary_dir'] = '/tmp/' + script_name + '/' + data_id # Directory in which to save the summaries
hyper_params['seed'] = None # Seed for the initialization of all random generators
hyper_params['single_thread'] = False # Force the execution on a single core in order to have a deterministic behavior

ddpg = DDPG( **hyper_params )


if len( sys.argv ) == 1 or sys.argv[1] != 'eval' :

	if len( sys.argv ) > 1 and sys.argv[1] == 'load' :
		if len( sys.argv ) > 2 :
			session_dir = sys.argv[2]
		ddpg.load_model( session_dir + '/session' )
		if not ddpg.load_replay_buffer( session_dir + '/replay_buffer.pkl' ) :
			print( 'Could not find %s: starting with an empty replay buffer.' % ( session_dir + '/replay_buffer.pkl' ) )


	np.random.seed( hyper_params['seed'] )

	training_env = ENV()
	eval_env = ENV()

	n_ep = 0
	Li = 0

	import time
	start = time.time()

	reward_graph = Monitor( titles='Average reward per trial', xlabel='trials', keep=False )

	with Loop_handler() as interruption :

		while not interruption() and n_ep < EP_MAX :

			s = training_env.reset()
			exploration = False

			for _ in range( EP_LEN ) :

				# Choose an action:
				if np.random.rand() < 0.1 :
					exploration = not exploration
					if exploration :
						a = np.random.uniform( -1, 1, hyper_params['a_dim'] )
				if not exploration :
					a = ddpg.get_action( s )
					#a = np.clip( ddpg.get_action( s ), -1, 1 )

				# Do one step:
				s2, r, terminal, _ = training_env.step( a )

				# Scale the reward:
				#r = r/10

				# Store the experience in the replay buffer:
				ddpg.replay_buffer.append(( s, a, r, terminal, s2 ))
				
				if terminal or interruption() :
					break

				s = s2

			if interruption() :
				break

			n_ep += 1

			# Train the networks (off-line):
			Li = ddpg.train( ITER_PER_EP )


			# Evaluate the policy:
			if Li != 0 and n_ep % EVAL_FREQ == 0 :
				s = eval_env.reset( store_data=True )
				for t in range( EP_LEN ) :
					s, _, done, _ = eval_env.step( ddpg.get_action( s ) )
					if done : break
				print( 'It %i | Ep %i | Li %+8.4f | ' % ( ddpg.n_iter, n_ep, Li ), end='' )
				eval_env.print_eval()
				sys.stdout.flush()
				ddpg.reward_summary( eval_env.get_Rt() )
				reward_graph.add_data( n_ep, eval_env.get_Rt() )


	end = time.time()
	print( 'Elapsed time: %.3f' % ( end - start ) )

	save_data = True
	answer = input( '\nSave network parameters in ' + session_dir + '? (y) ' )
	if answer.strip() != 'y' :
		answer = input( 'Where to store network parameters? (leave empty to discard data) ' )
		if answer.strip() :
			session_dir = answer
		else :
			save_data = False
	if save_data :
		os.makedirs( session_dir, exist_ok=True )
		ddpg.save_model( session_dir + '/session' )
		print( 'Parameters saved in %s.' % session_dir )
		answer = input( 'Save the replay buffer? (y) ' )
		if answer.strip() == 'y' :
			ddpg.save_replay_buffer( session_dir + '/replay_buffer.pkl' )
			print( 'Replay buffer saved as well.' )
		else :
			print( 'Replay buffer discarded.' )
	else :
		print( 'Data discarded.' )

else :


	if len( sys.argv ) > 2 :
		session_dir = sys.argv[2]
	ddpg.load_model( session_dir + '/session' )


	test_env = ENV( 180, store_data=True )
	s = test_env.get_obs()
	for t in range( EP_LEN ) :
		s, _, done, _ = test_env.step( ddpg.get_action( s ) )
		if done : break
	print( 'Trial result: ', end='' )
	test_env.print_eval()
	test_env.plot3D( ddpg.get_action, ddpg.get_V_value )
	test_env.plot_trial()
	test_env.show()


sess.close()
