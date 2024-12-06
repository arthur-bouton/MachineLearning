#!/usr/bin/env python3
""" 
Train a cart-pole with the Twin Delayed Deep Deterministic policy gradient (TD3) algorithm.

Start a new training by calling this script without argument.
When the algorithm has converged, send SIGINT to stop it and save all the variables and the content of the replay buffer by answering "y" when prompted to.
Then run the script with the word "eval" as first argument in order to evaluate the obtained policy.
The training can be took up with a saved set of variables and replay buffer by calling the script with the word "load" as first argument.

Author: Arthur Bouton [arthur.bouton@gadz.org]

Dependency:
tensorflow 2.3.1
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from TD3 import TD3
import numpy as np
import sys
import os
from cartpole import Cartpole
from looptools import Loop_handler, Monitor



# Identifier name for the data:
data_id = 'test1'

script_name = os.path.splitext( os.path.basename( __file__ ) )[0]

# Name of the files where to store the network parameters and replay buffer by default:
session_dir = './training_data/' + script_name + '/' + data_id

# Parameters for the training:
ENV = Cartpole # A class defining the environment
EP_LEN = 200 # Number of steps for one episode
ITER_PER_EP = 200 # Number of training iterations between each episode
EP_MAX = 100 # Maximal number of episodes for the training
EVAL_FREQ = 1 # Frequency of the policy evaluation

hyper_params = {}
hyper_params['s_dim'] = 4 # Dimension of the state space
hyper_params['a_dim'] = 1 # Dimension of the action space
hyper_params['state_scale'] = [ 1, 1, np.pi, 2*np.pi ] # A scalar or a vector to normalize the state
hyper_params['action_scale'] = None # A scalar or a vector to scale the actions
#hyper_params['actor_def'] = actor # The function defining the actor network
#hyper_params['critic_def'] = critic # The function defining the critic network
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


if len( sys.argv ) == 1 or sys.argv[1] != 'eval' :

	if len( sys.argv ) > 1 and sys.argv[1] == 'load' :
		if len( sys.argv ) > 2 :
			session_dir = sys.argv[2]
		td3.load( session_dir )
		if not td3.load_replay_buffer( session_dir + '/replay_buffer.pkl' ) :
			print( 'Could not find %s: starting with an empty replay buffer.' % ( session_dir + '/replay_buffer.pkl' ) )


	np.random.seed( hyper_params['seed'] )

	training_env = ENV()
	eval_env = ENV()

	n_ep = 0
	Q_loss = 0

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
						a = np.random.uniform( -1, 1, hyper_params['a_dim'] ).squeeze()
				if not exploration :
					a = td3.get_action( s )

				# Do one step:
				s2, r, ep_done, _ = training_env.step( a )

				# Store the experience in the replay buffer:
				td3.replay_buffer.append(( s, a, r, ep_done, s2 ))
				
				if ep_done or interruption() :
					break

				s = s2

			n_ep += 1


			if interruption() :
				break

			# Train the networks (off-line):
			Q_loss = td3.train( ITER_PER_EP )


			# Evaluate the policy:
			if Q_loss != 0 and n_ep % EVAL_FREQ == 0 :
				s = eval_env.reset( store_data=True )
				for t in range( EP_LEN ) :
					s, _, ep_done, _ = eval_env.step( td3.get_action( s ) )
					if ep_done : break
				print( 'It %i | Ep %i | LQ %+8.4f | ' % ( td3.n_iter, n_ep, Q_loss ), end='' )
				eval_env.print_eval()
				sys.stdout.flush()
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
		td3.save( session_dir )
		print( 'Parameters saved in %s.' % session_dir )
		answer = input( 'Save the replay buffer? (y) ' )
		if answer.strip() == 'y' :
			td3.save_replay_buffer( session_dir + '/replay_buffer.pkl' )
			print( 'Replay buffer saved as well.' )
		else :
			print( 'Replay buffer discarded.' )
	else :
		print( 'Data discarded.' )

else :


	if len( sys.argv ) > 2 :
		session_dir = sys.argv[2]
	td3.load( session_dir )


	test_env = ENV( ( 0, 180 ), store_data=True )
	s = test_env.get_obs()
	for t in range( EP_LEN ) :
		s, _, ep_done, _ = test_env.step( td3.get_action( s ) )
		if ep_done : break
	print( 'Trial result: ', end='' )
	test_env.print_eval()
	test_env.plot_trial()
	test_env.animate()
	#test_env.animate( 'cartpole_ddpg.gif' )
	test_env.show()
