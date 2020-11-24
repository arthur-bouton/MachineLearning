""" 
Implementation of the Deep Deterministic Policy Gradient (DDPG) algorithm [1] with TensorFlow for continuous-state and continuous-action spaces.

[1] Lillicrap, Timothy P., et al. "Continuous control with deep reinforcement learning." arXiv preprint arXiv:1509.02971 (2015).

Author: Arthur Bouton [arthur.bouton@gadz.org]

Dependency:
tensorflow 1.13.1
"""
import tensorflow as tf
import numpy as np
from collections import deque
import random
from tqdm import trange


def actor_network_def( states, a_dim ) :
	""" A feedforward neural network for the synthesis of the policy """

	s_dim = states.get_shape().as_list()[1]

	with tf.variable_scope( 'layer1' ) :
		n_units_1 = 400
		wmax = 1/np.sqrt( s_dim )
		bmax = 1/np.sqrt( s_dim )
		w1 = tf.get_variable( 'kernel', [s_dim, n_units_1], tf.float32, tf.initializers.random_uniform( -wmax, wmax ) )
		b1 = tf.get_variable( 'bias', [n_units_1], tf.float32, tf.initializers.random_uniform( -bmax, bmax ) )
		o1 = tf.add( tf.matmul( states, w1 ), b1 )
		a1 = tf.nn.relu( o1 )

	with tf.variable_scope( 'layer2' ) :
		n_units_2 = 300
		wmax = 1/np.sqrt( n_units_1 )
		bmax = 1/np.sqrt( n_units_1 )
		w2 = tf.get_variable( 'kernel', [n_units_1, n_units_2], tf.float32, tf.initializers.random_uniform( -wmax, wmax ) )
		b2 = tf.get_variable( 'bias', [n_units_2], tf.float32, tf.initializers.random_uniform( -bmax, bmax ) )
		o2 = tf.add( tf.matmul( a1, w2 ), b2 )
		a2 = tf.nn.relu( o2 )

	with tf.variable_scope( 'layer3' ) :
		wmax = 0.003
		bmax = 0.003
		w3 = tf.get_variable( 'kernel', [n_units_2, a_dim], tf.float32, tf.initializers.random_uniform( -wmax, wmax ) )
		b3 = tf.get_variable( 'bias', [a_dim], tf.float32, tf.initializers.random_uniform( -bmax, bmax ) )
		o3 = tf.add( tf.matmul( a2, w3 ), b3 )
		action = tf.nn.tanh( o3 )

	return action


def critic_network_def( states, actions ) :
	""" A feedforward neural network for the approximation of the Q-value """

	s_dim = states.get_shape().as_list()[1]
	a_dim = actions.get_shape().as_list()[1]

	with tf.variable_scope( 'layer1' ) :
		n_units_1 = 400
		wmax = 1/np.sqrt( s_dim )
		bmax = 1/np.sqrt( s_dim )
		w1 = tf.get_variable( 'kernel', [s_dim, n_units_1], tf.float32, tf.initializers.random_uniform( -wmax, wmax ) )
		b1 = tf.get_variable( 'bias', [n_units_1], tf.float32, tf.initializers.random_uniform( -bmax, bmax ) )
		o1 = tf.add( tf.matmul( states, w1 ), b1 )
		a1 = tf.nn.relu( o1 )

	with tf.variable_scope( 'layer2' ) :
		n_units_2 = 300
		wmax = 1/np.sqrt( n_units_1 + a_dim )
		bmax = 1/np.sqrt( n_units_1 + a_dim )
		w2 = tf.get_variable( 'kernel', [n_units_1 + a_dim, n_units_2], tf.float32, tf.initializers.random_uniform( -wmax, wmax ) )
		b2 = tf.get_variable( 'bias', [n_units_2], tf.float32, tf.initializers.random_uniform( -bmax, bmax ) )
		o2 = tf.add( tf.matmul( tf.concat( [ a1, actions ], 1 ), w2 ), b2 )
		a2 = tf.nn.relu( o2 )

	with tf.variable_scope( 'layer3' ) :
		wmax = 0.003
		bmax = 0.003
		w3 = tf.get_variable( 'kernel', [n_units_2, 1], tf.float32, tf.initializers.random_uniform( -wmax, wmax ) )
		b3 = tf.get_variable( 'bias', [1], tf.float32, tf.initializers.random_uniform( -bmax, bmax ) )
		Q_value = tf.add( tf.matmul( a2, w3 ), b3 )

	return Q_value


class DDPG() :
	"""
	Deep Deterministic Policy Gradient algorithm.

	Parameters
	----------
	s_dim : int
		Dimension of the state space.
	a_dim : int
		Dimension of the action space.
	state_scale : float or list of floats, optional, default: None
		A scalar or a vector to normalize the state.
	action_scale : float or list of floats, optional, default: None
		A scalar or a vector to scale the actions.
	gamma : float, optional, default: 0.99
		Discount factor applied to the reward.
	tau : float, optional, default: 1e-3
		Soft target update factor.
	buffer_size : int, optional, default: 1e6
		Maximal size of the replay buffer.
	minibatch_size : int, optional, default: 64
		Size of each minibatch.
	actor_lr : float, optional, default: 1e-4
		Learning rate of the actor network.
	critic_lr : float, optional, default: 1e-3
		Learning rate of the critic network.
	beta_L2 : float, optional, default: 0
		Ridge regularization coefficient.
	actor_def : function, optional, default: actor_network_def
		Function defining the actor network.
		It has to take the state tensor and the dimension of the
		action space as inputs and return the action tensor.
	critic_def : function, optional, default: critic_network_def
		Function defining the critic network.
		It has to take the state and action tensors as inputs and
		return the Q value tensor.
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
	# Fill the replay buffer with transitions:
	ddpg.replay_buffer.append(( state, action, reward, is_terminal, next_state ))

	# Train the networks:
	loss = ddpg.train( nb_iterations )

	# Infer the actions from the actor network:
	action = ddpg.get_action( state )

	"""

	def __init__( self, s_dim, a_dim, state_scale=None, action_scale=None,
	              gamma=0.99, tau=1e-3, buffer_size=1e6, minibatch_size=64, actor_lr=1e-4, critic_lr=1e-3, beta_L2=0,
				  actor_def=actor_network_def, critic_def=critic_network_def,
				  summary_dir=None, seed=None, sess=None, single_thread=False ) :

		self.gamma = gamma
		self.minibatch_size = minibatch_size
		self.summaries = summary_dir is not None

		self.n_iter = 0

		# Instantiation of the replay buffer:
		self.replay_buffer = deque( maxlen=int( buffer_size ) )
		random.seed( seed )

		######################
		# Building the graph #
		######################

		# Set the graph-level random seed:
		tf.set_random_seed( seed )

		self.states = tf.placeholder( tf.float32, [None, s_dim], 'States' )
		self.actions = tf.placeholder( tf.float32, [None, a_dim], 'Actions' )

		# Scaling of the inputs:
		if state_scale is not None :
			state_scale = tf.constant( state_scale, tf.float32, name='state_scale' )
			scaled_states = tf.divide( self.states, state_scale, 'scale_states' )
		else :
			scaled_states = self.states

		# Declaration of the actor network:
		with tf.variable_scope( 'Actor' ) :
			self.mu_actions = actor_def( scaled_states, a_dim )
			if action_scale is not None :
				action_scale = tf.constant( action_scale, tf.float32, name='action_scale' )
				self.mu_actions = tf.multiply( self.mu_actions, action_scale, 'scale_actions' )
			actor_params = tf.get_collection( tf.GraphKeys.TRAINABLE_VARIABLES, scope=tf.get_variable_scope().name )
		tf.identity( self.mu_actions, name='Actor_Output' )

		# Declaration of the critic network:
		with tf.variable_scope( 'Critic' ) :
			if action_scale is not None :
				scaled_actions = tf.divide( self.actions, action_scale, 'scale_actions' )
			else :
				scaled_actions = self.actions
			self.Q_value = critic_def( scaled_states, scaled_actions )
			critic_params = tf.get_collection( tf.GraphKeys.TRAINABLE_VARIABLES, scope=tf.get_variable_scope().name )

		# Declaration of the target networks:
		with tf.variable_scope( 'Target_Networks' ) :
			with tf.variable_scope( 'Target_Actor' ) :
				target_mu_actions = actor_def( scaled_states, a_dim )
				target_actor_params = tf.get_collection( tf.GraphKeys.TRAINABLE_VARIABLES, scope=tf.get_variable_scope().name )
			with tf.variable_scope( 'Target_Critic' ) :
				self.target_Q_value = critic_def( scaled_states, target_mu_actions )
				target_critic_params = tf.get_collection( tf.GraphKeys.TRAINABLE_VARIABLES, scope=tf.get_variable_scope().name )

		# Update of the target network parameters:
		with tf.name_scope( 'Update_Target_Networks' ) :
			sync_target_networks = [ tP.assign( P ) for P, tP in zip( actor_params + critic_params, target_actor_params + target_critic_params ) ]
			self.update_target_actor = [ tP.assign( P*tau + tP*( 1 - tau ) ) for P, tP in zip( actor_params, target_actor_params ) ]
			self.update_target_critic = [ tP.assign( P*tau + tP*( 1 - tau ) ) for P, tP in zip( critic_params, target_critic_params ) ]

		# Backpropagation in the critic network of the target errors:
		self.y = tf.placeholder( tf.float32, [None, 1], 'Targets' )
		with tf.name_scope( 'Critic_Backprop' ) :
			self.L = tf.losses.mean_squared_error( self.y, self.Q_value )
			if beta_L2 > 0 :
				with tf.name_scope( 'L2_regularization' ) :
					L2 = beta_L2*tf.reduce_mean( [ tf.nn.l2_loss( v ) for v in critic_params if 'kernel' in v.name ] )
					self.L += L2
			critic_optimizer = tf.train.AdamOptimizer( critic_lr )
			#self.train_critic = critic_optimizer.minimize( self.L, name='critic_backprop' )
			critic_grads_and_vars = critic_optimizer.compute_gradients( self.L, critic_params )
			self.train_critic = critic_optimizer.apply_gradients( critic_grads_and_vars, name='apply_backprop' )

		# Application of the deterministic policy gradient to the actor network:
		with tf.name_scope( 'Policy_Gradient' ) :
			gradQ_a = tf.gradients( self.Q_value, self.actions, name='gradQ_a' )
			gradQ_a_N = tf.divide( gradQ_a[0], tf.constant( minibatch_size, tf.float32, name='minibatch_size' ), 'normalize_over_batch' )
			policy_gradients = tf.gradients( self.mu_actions, actor_params, -gradQ_a_N, name='policy_gradients' )
			self.train_actor = tf.train.AdamOptimizer( actor_lr ).apply_gradients( zip( policy_gradients, actor_params ), name='apply_policy_gradients' )

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
		self.sess.run( sync_target_networks )
		self.saver = tf.train.Saver()

		# Create the summaries:
		if self.summaries :

			def param_histogram( params ) :
				for var in params :
					name = var.name.split( ':' )[0]
					tf.summary.histogram( name, var )

			param_histogram( actor_params )
			param_histogram( critic_params )
			self.wb_summary_op = tf.summary.merge_all()

			self.reward_eval = tf.placeholder( tf.float32, name='reward_eval' )
			reward_summary = tf.summary.scalar( 'Reward', self.reward_eval )
			self.reward_summary_op = tf.summary.merge( [ reward_summary ] )

			L_summary = tf.summary.scalar( 'L', self.L )
			cg_summaries = []
			with tf.name_scope( 'critic_gradient_norms' ) :
				for grad, var in critic_grads_and_vars :
					name = 'Critic_Gradient/' + var.name.split( '/', 1 )[1].split( ':' )[0]
					cg_summaries.append( tf.summary.scalar( name, tf.norm( grad ) ) )
			self.critic_summary_op = tf.summary.merge( [ L_summary ] + cg_summaries )

			ag_summaries = []
			with tf.name_scope( 'policy_gradient_norms' ) :
				for grad, var in zip( policy_gradients, actor_params ) :
					name = 'Policy_Gradient/' + var.name.split( '/', 1 )[1].split( ':' )[0]
					ag_summaries.append( tf.summary.scalar( name, tf.norm( grad ) ) )
			self.actor_summary_op = tf.summary.merge( ag_summaries )

			self.writer = tf.summary.FileWriter( summary_dir, self.sess.graph )

	def reward_summary( self, reward ) :

		if self.summaries :
			self.writer.add_summary( self.sess.run( self.reward_summary_op, {self.reward_eval: reward} ), self.n_iter )

	def sample_batch( self, batch_size ) :

		batch = random.sample( self.replay_buffer, min( len( self.replay_buffer ), batch_size ) )

		s_batch = np.array( [ _[0] for _ in batch ] )
		a_batch = np.array( [ _[1] if np.shape( _[1] ) else [ _[1] ] for _ in batch ] )
		r_batch = np.array( [ [ _[2] ] for _ in batch ] )
		t_batch = np.array( [ [ _[3] ] for _ in batch ] )
		s2_batch = np.array( [ _[4] for _ in batch ] )

		return s_batch, a_batch, r_batch, t_batch, s2_batch

	def train( self, iterations=1 ) :

		if len( self.replay_buffer ) < self.minibatch_size :
			return 0

		Lt = 0

		for _ in trange( iterations, desc='Training the networks', leave=False ) :

			self.n_iter += 1

			# Randomly pick samples in the replay buffer:
			s, a, r, terminal, s2 = self.sample_batch( self.minibatch_size )

			# Predict the future discounted rewards with the target critic network:
			target_q = self.sess.run( self.target_Q_value, {self.states: s2} )

			# Compute the targets for the Q-value:
			y = r + self.gamma*target_q*( 1 - terminal )

			# Optimize the critic network according to the targets:
			if self.summaries :
				L, _, critic_summary = self.sess.run( [ self.L, self.train_critic, self.critic_summary_op ], {self.states: s, self.actions: a, self.y: y} )
				self.writer.add_summary( critic_summary, self.n_iter )
			else :
				L, _ = self.sess.run( [ self.L, self.train_critic ], {self.states: s, self.actions: a, self.y: y} )
			Lt += L

			# Apply the sample gradient to the actor network:
			mu_a = self.sess.run( self.mu_actions, {self.states: s} )
			if self.summaries :
				_, actor_summary = self.sess.run( [ self.train_actor, self.actor_summary_op ], {self.states: s, self.actions: mu_a} )
				self.writer.add_summary( actor_summary, self.n_iter )
			else :
				self.sess.run( self.train_actor, {self.states: s, self.actions: mu_a} )

			# Update the target networks:
			self.sess.run( self.update_target_actor )
			self.sess.run( self.update_target_critic )

		if self.summaries and self.wb_summary_op is not None :
			self.writer.add_summary( self.sess.run( self.wb_summary_op ), self.n_iter )
			#self.writer.flush()

		return Lt/iterations
	
	def get_action( self, s ) :
		mu_a = self.sess.run( self.mu_actions, {self.states: s[np.newaxis, :] if s.ndim < 2 else s} )
		return mu_a.squeeze()
	
	def get_Q_value( self, s, a ) :
		Q_value = self.sess.run( self.Q_value, {self.states: s[np.newaxis, :] if s.ndim < 2 else s,
		                                        self.actions: a[:, np.newaxis] if isinstance( a, np.ndarray ) and a.ndim < 2 else a} )
		return Q_value.squeeze()
	
	def get_V_value( self, s ) :
		mu_a = self.sess.run( self.mu_actions, {self.states: s[np.newaxis, :] if s.ndim < 2 else s} )
		V_value = self.sess.run( self.Q_value, {self.states: s[np.newaxis, :] if s.ndim < 2 else s, self.actions: mu_a} )
		return V_value.squeeze()
	
	def save_model( self, filename ) :
		self.saver.save( self.sess, filename )
	
	def load_model( self, filename ) :
		self.saver.restore( self.sess, filename )

	def save_replay_buffer( self, filename ) :
		with open( filename, 'wb' ) as f :
			import pickle
			pickle.dump( self.replay_buffer, f )

	def load_replay_buffer( self, filename ) :
		try :
			with open( filename, 'rb' ) as f :
				import pickle
				temp_buf = pickle.load( f )
			self.replay_buffer = temp_buf
			return True
		except IOError :
			return False

	def __enter__( self ) :
		return self

	def __exit__( self, type, value, traceback ) :
		self.sess.close()
