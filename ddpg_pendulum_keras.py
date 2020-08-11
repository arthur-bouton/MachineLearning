#!/usr/bin/env python3
""" 
Train a simple pendulum with the Deep Deterministic Policy Gradient (DDPG) algorithm.

Start a new training by calling this script without argument.
When the algorithm has converged, send SIGINT to stop it and save the model by answering "y" when prompted to.
Then run the script with the word "eval" as first argument in order to evaluate the obtained policy.
The training can be took up with a saved model by calling the script with the word "load" as first argument.

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

# Name of the files where to store the network parameters:
session_dir = './training_data/' + script_name + '/' + data_id
session_files = session_dir + '/session'

# Parameters for the training:
ENV = Pendulum # A class defining the environment
EP_LEN = 100 # Number of steps for one episode
EP_MAX = 2000 # Maximal number of episodes for the training
ITER_PER_EP = 200 # Number of training iterations between each episode
S_DIM = 2 # Dimension of the state space
A_DIM = 1 # Dimension of the action space
STATE_SCALE = [ np.pi, 2*np.pi ] # A scalar or a vector to normalize the state
ACTION_SCALE = None # A scalar or a vector to scale the actions
GAMMA = 0.7 # Discount factor of the reward
TAU = 0.001 # Soft target update factor
BUFFER_SIZE = 100000 # Maximal size of the replay buffer
MINIBATCH_SIZE = 128 # Size of each minibatch
ACTOR_LR = 0.0001 # Learning rate of the actor network
CRITIC_LR = 0.001 # Learning rate of the critic network
BETA_L2 = 0 # Ridge regularization coefficient
#ALPHA_SAMPLING = 1 # Exponent interpolating between uniform sampling (0) and greedy prioritization (1) (DDPG_PER only)
#BETA_IS = 0 # Exponent of the importance-sampling weights (if 0, no importance sampling) (DDPG_PER only)
SUMMARY_DIR = None # No summaries
#SUMMARY_DIR = '/tmp/' + script_name + '/' + data_id # Directory where to save summaries
SEED = None # Random seed for the initialization of all random generators
SINGLE_THREAD = False # Force the execution on a single core in order to have a deterministic behavior

ddpg = DDPG( S_DIM, A_DIM, STATE_SCALE, ACTION_SCALE, GAMMA, TAU, BUFFER_SIZE, MINIBATCH_SIZE, ACTOR_LR, CRITIC_LR, BETA_L2,
		   actor_def=actor, critic_def=critic,
		   #alpha_sampling=ALPHA_SAMPLING, beta_IS=BETA_IS, # (DDPG_PER only)
		   summary_dir=SUMMARY_DIR, seed=SEED, single_thread=SINGLE_THREAD, sess=sess )


if len( sys.argv ) == 1 or sys.argv[1] != 'eval' :

	if len( sys.argv ) > 1 and sys.argv[1] == 'load' :
		if len( sys.argv ) > 2 :
			ddpg.load_model( sys.argv[2] + '/session' )
		else :
			ddpg.load_model( session_files )


	np.random.seed( SEED )

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
			expl = False

			for _ in range( EP_LEN ) :

				# Choose an action:
				if np.random.rand() < 0.1 :
					if expl :
						expl = False
					else :
						expl = True
						a = np.random.uniform( -1, 1, A_DIM )
				if not expl :
					a = ddpg.get_action( s )
					#a = np.clip( ddpg.get_action( s ), -1, 1 )

				# Do one step:
				s2, r, terminal, _ = training_env.step( a )

				# Scale the reward:
				#r = r/10

				# Store the experience in the replay buffer:
				ddpg.replay_buffer.append(( s, a, r, terminal, s2 ))

				# When there is enough samples, train the networks (on-line):
				#if len( ddpg.replay_buffer ) > ddpg.minibatch_size :
					#Li = ddpg.train()
				
				if terminal or interruption() :
					break

				s = s2

			if interruption() :
				break

			n_ep += 1

			# When there is enough samples, train the networks (off-line):
			if len( ddpg.replay_buffer ) >= ddpg.minibatch_size :
				Li = ddpg.train( ITER_PER_EP )


			# Evaluate the policy:
			if n_ep % 1 == 0 :
				s = eval_env.reset( store_data=True )
				for t in range( EP_LEN ) :
					s, _, done, _ = eval_env.step( ddpg.get_action( s ) )
					if done : break
				print( 'It %i | Ep %i | Li %+8.4f | ' % ( ddpg.n_iter, n_ep, Li ), end='', flush=True )
				eval_env.print_eval()
				sys.stdout.flush()
				ddpg.reward_summary( eval_env.get_Rt() )
				reward_graph.add_data( n_ep, eval_env.get_Rt() )


	end = time.time()
	print( 'Elapsed time: %.3f' % ( end - start ) )

	answer = input( '\nSave network parameters in ' + session_dir + '? (y) ' )
	if answer.strip() == 'y' :
		os.makedirs( session_dir, exist_ok=True )
		ddpg.save_model( session_files )
		print( 'Parameters saved.' )
	else :
		answer = input( 'Where to store network parameters? (leave empty to discard data) ' )
		if answer.strip() :
			os.makedirs( answer, exist_ok=True )
			ddpg.save_model( answer + '/session' )
			print( 'Parameters saved in %s.' % answer )
		else :
			print( 'Data discarded.' )

else :


	if len( sys.argv ) > 2 :
		ddpg.load_model( sys.argv[2] + '/session' )
	else :
		ddpg.load_model( session_files )


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
