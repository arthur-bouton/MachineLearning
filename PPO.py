"""
Multithreaded implementation of the Proximal Policy Optimization (PPO) algorithm [1] with TensorFlow.

Distribute workers to parallel threads in order to speed up the collection of data from the current policy at each step.

[1] Schulman, John, et al. "Proximal policy optimization algorithms." arXiv preprint arXiv:1707.06347 (2017).

Author: Arthur Bouton [arthur.bouton@gadz.org]

Dependencies:
tensorflow r1.13.1
"""
import tensorflow as tf
import numpy as np
import random
import collections
from tqdm import trange


class PPO() :
	"""
	Proximal Policy Optimization algorithm.

	Parameters
	----------
	s_dim : int
		Dimension of the state space.
	a_dim : int
		Dimension of the action space.
	actor_def : function
		Function defining the actor network.
		It has to take the state tensor and the dimension of the
		action space as inputs and return the both the mean and
		the standard deviation of the normal distribution to use.
	critic_def : function
		Function defining the critic network.
		It has to take the state tensor as an input and
		return the V value tensor.
	state_scale : float or list of floats, optional, default: None
		A scalar or a vector to normalize the state.
	action_scale : float or list of floats, optional, default: None
		A scalar or a vector to scale the actions.
	gamma : float, optional, default: 0.99
		Discount factor applied to the reward.
	gae_lambda : float, optional, default: 0.95
		Generalized advantage estimation parameter.
		If 0, the returns are the difference of each reward with only
		the V value estimation of the next step.
		If 1, the returns are comprised only of the discounted rewards
		from the trial.
	epsilon : float, optional, default: 0.2
		Surrogate objective clipping parameter.
	learning_rate : float, optional, default: 1e-4
		Learning rate of the gradient descent optimization algorithm.
	vf_coeff : float, optional, default: 1
		Critic loss coefficient in the objective function.
	entropy_coeff : float, optional, default: 0.01
		Coefficient of the entropy bonus in the objective function
		in order to ensure sufficient exploration.
	minibatch_size : int, optional, default: 64
		Size of each minibatch.
	epochs : int, optional, default: 200
		Number of iteration at each update.
	summary_dir : str, optional, default: None
		Directory in which to save the summaries.
		If None, no summaries are created.
	seed : int, optional, default: None
		Random seed for the initialization of random generators.
	sess : tf.Session, optional, default: None
		A TensorFlow session already initialized.
		If None, a new session is created.
	single_thread : bool, optional, default: False
		Whether to force the execution on a single core in order to
		have a deterministic behavior (sess=None only).

	Examples
	--------
	# Sample actions from the stochastic policy:
	action = ppo.stoch_action( state )

	# Add consecutive transitions from the current stochastic policy:
	ppo.add_transitions(( state, action, reward, is_terminal, next_state ))

	# Once the episode over, compute the generalized advantage estimation
	and store the data before starting a new episode or training the networks:
	ppo.process_episode()

	# Train the networks:
	loss = ppo.train()

	# Infer the optimal actions from the actor network:
	action = ppo.best_action( state )

	# Use multiple worker threads running in background, with 'trial_generator'
	# a generator returning the successive transitions of a single episode:
	with ppo.workers( nb_of_worker_threads, trial_generator, nb_of_episodes_per_batch ) :
		for iteration in range( nb_of_iterations ) :

			# Gather new data from the workers:
			nb_of_episodes_from_beginning, nb_of_samples_collected = ppo.rollout()

			# Train the networks:
			loss = ppo.train()

	"""

	def __init__( self, s_dim, a_dim, actor_def, critic_def, state_scale=None, action_scale=None,
	              gamma=0.99, gae_lambda=0.95, epsilon=0.2, learning_rate=1e-4, vf_coeff=1, entropy_coeff=0.01,
				  minibatch_size=64, epochs=200,
				  summary_dir=None, seed=None, sess=None, single_thread=False ) :

		self.gamma = gamma
		self.gae_lambda = gae_lambda
		self.minibatch_size = minibatch_size
		self.epochs = epochs
		self.summaries = summary_dir is not None

		self.n_iter = 0
		self.n_ep = 0
		self.experience = [ [] for _ in range( 4 ) ]
		self.threads = []

		# Set the random seed for the minibatch sampling:
		random.seed( seed )

		######################
		# Building the graph #
		######################

		# Set the graph-level random seed:
		tf.set_random_seed( seed )
		
		self.states = tf.placeholder( tf.float32, [None, s_dim], 'States' )
		self.actions = tf.placeholder( tf.float32, [None, a_dim], 'Actions' )
		self.returns = tf.placeholder( tf.float32, [None, 1], 'Returns' )
		self.values = tf.placeholder( tf.float32, [None, 1], 'Values' )

		# Scaling of the inputs:
		if state_scale is not None :
			state_scale = tf.constant( state_scale, tf.float32, name='state_scale' )
			scaled_states = tf.divide( self.states, state_scale, 'scale_states' )
		else :
			scaled_states = self.states

		# Declaration of the actor network:
		with tf.variable_scope( 'Pi' ) :
			self.pi_mu, self.pi_sigma = actor_def( scaled_states, a_dim )
			if action_scale is not None :
				action_scale = tf.constant( action_scale, tf.float32, name='action_scale' )
				self.pi_mu = tf.multiply( self.pi_mu, action_scale, 'scale_actions' )
				self.pi_sigma = tf.multiply( self.pi_sigma, action_scale, 'scale_stddev' )
			pi_params = tf.get_collection( tf.GraphKeys.TRAINABLE_VARIABLES, scope=tf.get_variable_scope().name )
			stoch_pi = tf.distributions.Normal( self.pi_mu, self.pi_sigma, allow_nan_stats=False )

		# Declaration of the non-trained reference actor network to be compared with at each update:
		with tf.variable_scope( 'Pi_old' ) :
			oldpi_mu, oldpi_sigma = actor_def( scaled_states, a_dim )
			if action_scale is not None :
				oldpi_mu = tf.multiply( oldpi_mu, action_scale, 'scale_actions' )
				oldpi_sigma = tf.multiply( oldpi_sigma, action_scale, 'scale_stddev' )
			oldpi_params = tf.get_collection( tf.GraphKeys.TRAINABLE_VARIABLES, scope=tf.get_variable_scope().name )
			stoch_oldpi = tf.distributions.Normal( oldpi_mu, oldpi_sigma, allow_nan_stats=False )

		# Remove the reference actor network parameters from the trainable variables:
		for var in oldpi_params :
			tf.get_collection_ref( tf.GraphKeys.TRAINABLE_VARIABLES ).remove( var )

		# Sample one stochastic action from the current policy:
		self.sample_pi_op = stoch_pi.sample( 1 )

		# Synchronize the reference actor network:
		with tf.name_scope( 'Update_Pi_old' ) :
			self.update_oldpi_op = [ oldp.assign( p ) for p, oldp in zip( pi_params, oldpi_params ) ]

		# Declaration of the critic network:
		with tf.variable_scope( 'Critic' ) :
			self.V_value = critic_def( scaled_states )

		with tf.name_scope( 'Objective' ) :
			# Compute the clipped surrogate objective:
			ratios = tf.divide( stoch_pi.prob( self.actions ), stoch_oldpi.prob( self.actions ) + 1e-10 )
			clipped_ratios = tf.clip_by_value( ratios, 1. - epsilon, 1. + epsilon )
			advantages = self.returns - self.values
			L_clip = tf.minimum( ratios*advantages, clipped_ratios*advantages )

			# Target errors to backpropagate into the critic network:
			L_vf = tf.losses.mean_squared_error( self.returns, self.V_value )

			# Addition of an entropy bonus to ensure sufficient exploration:
			entropy = stoch_pi.entropy()

			# Combination of the three terms and normalization over each minibatch:
			self.L = -tf.reduce_mean( L_clip - vf_coeff*L_vf + entropy_coeff*entropy )

			self.train_op = tf.train.AdamOptimizer( learning_rate=learning_rate ).minimize( self.L )

		#######################
		# Setting the session #
		#######################

		if sess is not None :
			self.sess = sess
		else :
			if single_thread :
				sess_config = tf.ConfigProto( intra_op_parallelism_threads=1, inter_op_parallelism_threads=1 )
				self.sess = tf.Session( config=sess_config )
			else :
				self.sess = tf.Session()

		# Initialize variables and saver:
		self.sess.run( tf.global_variables_initializer() )
		self.saver = tf.train.Saver()

		# Create the summaries:
		if self.summaries :

			def param_histogram( params ) :
				for var in params :
					name = var.name.split( ':' )[0]
					tf.summary.histogram( name, var )

			critic_params = tf.get_collection( tf.GraphKeys.TRAINABLE_VARIABLES, scope='Critic' )
			param_histogram( pi_params )
			param_histogram( critic_params )
			self.wb_summary_op = tf.summary.merge_all()

			self.reward_eval = tf.placeholder( tf.float32, name='reward_eval' )
			reward_summary = tf.summary.scalar( 'Reward', self.reward_eval )
			self.reward_summary_op = tf.summary.merge( [ reward_summary ] )

			L_summary = tf.summary.scalar( 'L', self.L )
			self.loss_summary_op = tf.summary.merge( [ L_summary ] )

			self.writer = tf.summary.FileWriter( summary_dir, self.sess.graph )

	def reward_summary( self, reward ) :

		if self.summaries :
			self.writer.add_summary( self.sess.run( self.reward_summary_op, {self.reward_eval: reward} ), self.n_iter )

	def train( self ) :

		if len( self.experience[2] ) == 0 :
			raise RuntimeError( 'No new experience has been added' )

		self.n_iter += 1

		# Synchronize the reference actor network:
		self.sess.run( self.update_oldpi_op )

		# Unfold the experience obtained with the current policy:
		s, a, r, v = map( np.vstack, self.experience )

		# Train the networks by minibatches:
		Lt = 0
		for epoch in trange( self.epochs, desc='Training the networks', leave=False ) :
			indices = random.sample( range( len( r ) ), self.minibatch_size )
			if self.summaries :
				L, _, loss_summary = self.sess.run( [ self.L, self.train_op, self.loss_summary_op ], {self.states: s[indices], self.actions: a[indices], self.returns: r[indices], self.values: v[indices]} )
				self.writer.add_summary( loss_summary, self.n_iter*self.epochs + epoch )
			else :
				L, _ = self.sess.run( [ self.L, self.train_op ], {self.states: s[indices], self.actions: a[indices], self.returns: r[indices], self.values: v[indices]} )
			Lt += L

		# Discard the experience data:
		self.experience = [ [] for _ in range( 4 ) ]

		if self.summaries and self.wb_summary_op is not None :
			self.writer.add_summary( self.sess.run( self.wb_summary_op ), self.n_iter )
			#self.writer.flush()

		return Lt

	def _generalized_advantage_estimation( self, values, next_value, rewards, masks ) :
		values = values + [next_value]
		gae = 0
		returns = []
		for t in reversed( range( len( rewards ) ) ) :
			delta = rewards[t] + self.gamma*masks[t]*values[t+1] - values[t]
			gae = delta + self.gamma*self.gae_lambda*masks[t]*gae
			returns.insert( 0, gae + values[t] )

		#T = 20
		#N = len( rewards )
		#deltas = []
		#gammalambdas = [ 1 ]
		#gl = self.gamma*self.gae_lambda
		#for t in range( N ) :
			#deltas.append( rewards[t] + self.gamma*masks[t]*values[t+1] - values[t] )
			#gammalambdas.append( gl*gammalambdas[-1] )
		#returns = []
		#for t in range( N ) :
			#end = min( T, t - N )
			#returns.append( sum( [ c*d for c, d in zip( gammalambdas[0:end], deltas[t:t+end] ) ] ) )

		return returns

	def _worker( self, worker_id, trial_generator, episodes_per_batch=1, samples_per_batch=None ) :

		self.rolling_out.wait()

		while not self.worker_termination.is_set() :

			if not self.no_more_ep :

				self.ep_count += 1
				self.n_ep += 1

				if episodes_per_batch is not None and self.ep_count >= episodes_per_batch :
					self.no_more_ep = True

				states, actions, rewards, values, masks = [], [], [], [], []

				for transitions in trial_generator( worker_id ) :

					if len( transitions ) == 5 and not isinstance( transitions[2], collections.Iterable ) :
						transitions = [ transitions ]

					for s, a, r, ep_done, s_next in transitions :

						# Store the data for the episode:
						states.append( s )
						actions.append( a )
						rewards.append( r )
						values.append( self.get_value( s ) )
						masks.append( int( not ep_done ) )

						self.samp_count += 1

					if samples_per_batch is not None and self.samp_count >= samples_per_batch or ep_done :
						break

				# Compute the generalized advantage estimation for the episode:
				next_value = self.get_value( s_next )
				returns = self._generalized_advantage_estimation( values, next_value, rewards, masks )

				# Store the experience:
				with self.experience_mutex :
					for i, new_data in enumerate( ( states, actions, returns, values ) ) :
						self.experience[i].extend( new_data )

			if self.no_more_ep or samples_per_batch is not None and self.samp_count >= samples_per_batch :
				# Stop starting new trials:
				self.rolling_out.clear()

				# Wait for other workers to store their data:
				self.worker_count -= 1
				if self.worker_count <= 0 :
					# When all workers have stored their data, signal the end of the roll-out:
					self.rollout_over.set()

				# Wait for a new roll-out to start:
				self.rolling_out.wait()

	def start_workers( self, n_workers, trial_generator, episodes_per_batch=1, samples_per_batch=None ) :

		if episodes_per_batch is None and samples_per_batch is None :
			raise RuntimeError( "'episodes_per_batch' and 'samples_per_batch' cannot be both None" )

		import threading

		# Create the events to control the worker threads (flags are initially cleared):
		self.rolling_out, self.rollout_over = threading.Event(), threading.Event()

		# Create the event requesting the termination of the worker threads:
		self.worker_termination = threading.Event()

		# Create a mutex to protect the shared experience list when it is filled by worker threads:
		self.experience_mutex = threading.Lock()

		# Start the worker threads:
		for worker_id in range( n_workers ) :
			t = threading.Thread( target=self._worker, args=( worker_id + 1, trial_generator, episodes_per_batch, samples_per_batch ) )
			t.start()
			self.threads.append( t )

		return self.threads

	def stop_workers( self ) :

		# Flag the request for termination:
		self.worker_termination.set()

		# Free blocked workers:
		self.rolling_out.set()

		# Wait for every thread to end:
		for t in self.threads :
			t.join()

	def workers( self, n_workers, trial_generator, episodes_per_batch=1, samples_per_batch=None ) :
		""" Return a context manager that starts and stops the multithreaded workers """

		class _workers_context_manager() :

			def __init__( self, ppo, n_workers, trial_generator, episodes_per_batch, samples_per_batch ) :
				self.ppo = ppo
				self.n_workers = n_workers
				self.trial_gen = trial_generator
				self.episodes_per_batch = episodes_per_batch
				self.samples_per_batch = samples_per_batch

			def __enter__( self ) :
				return self.ppo.start_workers( self.n_workers, self.trial_gen, self.episodes_per_batch, self.samples_per_batch )

			def __exit__( self, type, value, traceback ) :
				self.ppo.stop_workers()

		return _workers_context_manager( self, n_workers, trial_generator, episodes_per_batch, samples_per_batch )

	def rollout( self ) :

		if len( self.threads ) == 0 :
			raise RuntimeError( 'There is no worker thread running' )

		self.ep_count = 0
		self.samp_count = 0
		self.no_more_ep = False
		self.worker_count = len( self.threads )
		self.rollout_over.clear()

		# Start the next roll-out:
		self.rolling_out.set()

		# Wait for a new batch of data:
		self.rollout_over.wait()

		return self.n_ep, self.samp_count
	
	def add_transitions( self, transitions ) :

		if not hasattr( self, 'ep_buf' ) :
			self.ep_buf = { 's': [], 'a': [], 'r': [], 'v': [], 'm': [], 's_next': None }

		if len( transitions ) == 5 and not isinstance( transitions[2], collections.Iterable ) :
			transitions = [ transitions ]

		for s, a, r, ep_done, self.ep_buf['s_next'] in transitions :

			# Store the data for the episode:
			self.ep_buf['s'].append( s )
			self.ep_buf['a'].append( a )
			self.ep_buf['r'].append( r )
			self.ep_buf['v'].append( self.get_value( s ) )
			self.ep_buf['m'].append( int( not ep_done ) )

	def process_episode( self ) :

		# Compute the generalized advantage estimation for the episode:
		next_value = self.get_value( self.ep_buf['s_next'] )
		returns = self._generalized_advantage_estimation( self.ep_buf['v'], next_value, self.ep_buf['r'], self.ep_buf['m'] )

		# Store the experience:
		for i, new_data in enumerate( ( self.ep_buf['s'], self.ep_buf['a'], returns, self.ep_buf['v'] ) ) :
			self.experience[i].extend( new_data )

		# Discard the episode buffer:
		self.ep_buf = { 's': [], 'a': [], 'r': [], 'v': [], 'm': [], 's_next': None }

	def stoch_action( self, s ) :
		a = self.sess.run( self.sample_pi_op, {self.states: s[np.newaxis, :] if s.ndim < 2 else s} )
		return a.squeeze()

	def best_action( self, s, return_stddev=False ) :
		mu, sigma = self.sess.run( [ self.pi_mu, self.pi_sigma ], {self.states: s[np.newaxis, :] if s.ndim < 2 else s} )
		if return_stddev :
			return mu.squeeze(), sigma.squeeze()
		return mu.squeeze()

	def get_value( self, s ) :
		v = self.sess.run( self.V_value, {self.states: s[np.newaxis, :] if s.ndim < 2 else s} )
		return v.squeeze()
	
	def save( self, filename ) :
		self.saver.save( self.sess, filename )
	
	def load( self, filename ) :
		self.saver.restore( self.sess, filename )

	def __enter__( self ) :
		return self

	def __exit__( self, type, value, traceback ) :
		self.sess.close()
