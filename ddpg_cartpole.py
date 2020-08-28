#!/usr/bin/env python3
""" 
Train a cartpole with the Deep Deterministic Policy Gradient (DDPG) algorithm.

Start a new training by calling this script without argument.
When the algorithm has converged, send SIGINT to stop it and save the model and the content of the replay buffer by answering "y" when prompted to.
Then run the script with the word "eval" as first argument in order to evaluate the obtained policy.
The training can be took up with a saved model and replay buffer by calling the script with the word "load" as first argument.

Author: Arthur Bouton [arthur.bouton@gadz.org]

Dependency:
keras 2.3.1
"""
from DDPG_vanilla import DDPG
#from DDPG_PER import DDPG
import numpy as np
import sys
import os
from cartpole import Cartpole
from looptools import Loop_handler, Monitor


from keras.layers import Dense, Concatenate


# Actor network:
def actor( states, a_dim ) :

	x = Dense( 200, activation='relu' )( states )
	x = Dense( 200, activation='relu' )( x )
	action = Dense( a_dim, activation='tanh' )( x )

	return action


# Critic network:
def critic( states, actions ) :

	x = Concatenate()( [ states, actions ] )
	x = Dense( 200, activation='relu' )( x )
	x = Dense( 200, activation='relu' )( x )
	Q_value = Dense( 1, activation='linear' )( x )

	return Q_value


# Identifier name for the data:
data_id = 'test1'

script_name = os.path.splitext( os.path.basename( __file__ ) )[0]

# Name of the files where to store the network parameters and replay buffer by default:
session_dir = './training_data/' + script_name + '/' + data_id

# Parameters for the training:
ENV = Cartpole # A class defining the environment
EP_LEN = 200 # Number of steps for one episode
ITER_PER_EP = 400 # Number of training iterations between each episode
EP_MAX = 1000 # Maximal number of episodes for the training
EVAL_FREQ = 1 # Frequency of the policy evaluation

hyper_params = {}
hyper_params['s_dim'] = 4 # Dimension of the state space
hyper_params['a_dim'] = 1 # Dimension of the action space
hyper_params['state_scale'] = [ 1, 1, np.pi, 2*np.pi ] # A scalar or a vector to normalize the state
hyper_params['action_scale'] = None # A scalar or a vector to scale the actions
hyper_params['actor_def'] = actor # The function defining the actor network
hyper_params['critic_def'] = critic # The function defining the critic network
hyper_params['gamma'] = 0.9 # Discount factor applied to the reward
hyper_params['tau'] = 1e-3 # Soft target update factor
hyper_params['buffer_size'] = 1e4 # Maximal size of the replay buffer
hyper_params['minibatch_size'] = 64 # Size of each minibatch
hyper_params['actor_lr'] = 1e-4 # Learning rate of the actor network
hyper_params['critic_lr'] = 1e-3 # Learning rate of the critic network
hyper_params['beta_L2'] = 0 # Ridge regularization coefficient
#hyper_params['alpha_sampling'] = 1 # Exponent interpolating between a uniform sampling (0) and a greedy prioritization (1) (DDPG_PER only)
#hyper_params['beta_IS'] = 1 # Exponent of the importance-sampling weights (if 0, no importance sampling) (DDPG_PER only)
#hyper_params['summary_dir'] = '/tmp/' + script_name + '/' + data_id # Directory in which to save the summaries
hyper_params['seed'] = None # Random seed for the initialization of all random generators
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
	L = 0

	reward_graph = Monitor( titles='Average reward per trial', xlabel='trials', keep=False )

	import time
	start = time.time()

	with Loop_handler() as interruption :

		while not interruption() and n_ep < EP_MAX :


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
					a = ddpg.get_action( s )

				# Do one step:
				s2, r, terminal, _ = training_env.step( a )

				# Scale the reward:
				#r = r/10

				# Store the experience in the replay buffer:
				ddpg.replay_buffer.append(( s, a, r, terminal, s2 ))
				
				if terminal or interruption() :
					break

				s = s2

			n_ep += 1


			if interruption() :
				break

			# Train the networks (off-line):
			L = ddpg.train( ITER_PER_EP )


			# Evaluate the policy:
			if L != 0 and n_ep % EVAL_FREQ == 0 :
				s = eval_env.reset( store_data=True )
				for t in range( EP_LEN ) :
					s, _, done, _ = eval_env.step( ddpg.get_action( s ) )
					if done : break
				print( 'It %i | Ep %i | L %+8.4f | ' % ( ddpg.n_iter, n_ep, L ), end='' )
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


	test_env = ENV( ( 0, 180 ), store_data=True )
	s = test_env.get_obs()
	for t in range( EP_LEN ) :
		s, _, done, _ = test_env.step( ddpg.get_action( s ) )
		if done : break
	print( 'Trial result: ', end='' )
	test_env.print_eval()
	test_env.plot_trial()
	test_env.animate()
	#test_env.animate( 'cartpole_ddpg.gif' )
	test_env.show()


ddpg.sess.close()
