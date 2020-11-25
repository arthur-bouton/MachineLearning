""" 
Implementation of the Soft Actor-Critic (SAC) algorithm with automated entropy temperature adjustment [1] for continuous-state and continuous-action spaces using TensorFlow 2.

[1] Haarnoja, Tuomas, et al. "Soft actor-critic algorithms and applications." arXiv preprint arXiv:1812.05905 (2018).

Author: Arthur Bouton [arthur.bouton@gadz.org]

Dependency:
tensorflow 2.3.1
"""
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from collections import deque
import random
from tqdm import trange


from tensorflow import keras
from tensorflow.keras import layers


# Default actor network:
def actor_model_def( s_dim, a_dim ) :

	states = keras.Input( shape=s_dim )

	x = layers.Dense( 256, activation='relu' )( states )
	x = layers.Dense( 256, activation='relu' )( x )

	mu = layers.Dense( a_dim, activation='linear' )( x )

	x = layers.Dense( 256, activation='relu' )( states )
	x = layers.Dense( 256, activation='relu' )( x )

	sigma = layers.Dense( a_dim, activation='softplus' )( x )

	return keras.Model( states, [ mu, sigma ], name='actor' )


# Default critic network:
def critic_model_def( s_dim, a_dim, name=None ) :

	states  = keras.Input( shape=s_dim )
	actions = keras.Input( shape=a_dim )

	x = layers.Concatenate()( [ states, actions ] )
	x = layers.Dense( 256, activation='relu' )( x )
	x = layers.Dense( 256, activation='relu' )( x )
	Q_value = layers.Dense( 1, activation='linear' )( x )

	return keras.Model( [ states, actions ], Q_value, name=name )


class SAC :
	"""
	Soft Actor-Critic algorithm.

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
	target_entropy : negative float, optional, default: None
		Desired target entropy of the policy.
	tau : float, optional, default: 1e-3
		Soft target update factor.
	buffer_size : int, optional, default: 1e6
		Maximal size of the replay buffer.
	minibatch_size : int, optional, default: 64
		Size of each minibatch.
	learning_rate : float, optional, default: 3e-4
		Default learning rate used for all the networks.
	actor_lr : float, optional, default: None
		Learning rate to use for the optimization of the actor network.
		If None, learning_rate is used.
	critic_lr : float, optional, default: None
		Learning rate to use for the optimization of the critic networks.
		If None, learning_rate is used.
	alpha_lr : float, optional, default: None
		Learning rate to use for the optimization of the entropy temperature.
		If None, learning_rate is used.
	alpha0 : float, optional, default: 0.7
		Initial value of the entropy temperature.
	actor_def : function, optional, default: actor_model_def
		Function defining the actor model.
		It has to take the dimension of the state and the action spaces
		as inputs and return a Keras model.
	critic_def : function, optional, default: critic_model_def
		Function defining the critic model.
		It has to take the dimension of the state and the action spaces
		as inputs and return a Keras model.
	seed : int, optional, default: None
		Random seed for the initialization of random generators.

	Examples
	--------
	# Sample actions from the stochastic policy:
	action = sac.stoch_action( state )

	# Fill the replay buffer with transitions:
	sac.replay_buffer.append(( state, action, reward, is_terminal, next_state ))

	# Train the networks:
	loss = sac.train( nb_iterations )

	# Infer the best actions from the current policy:
	action = sac.best_action( state )

	"""

	def __init__( self, s_dim, a_dim, state_scale=None, action_scale=None,
	              gamma=0.99, target_entropy=None, tau=5e-3, buffer_size=1e6, minibatch_size=256,
				  learning_rate=3e-4, actor_lr=None, critic_lr=None, alpha_lr=None,
				  alpha0=0.7, actor_def=actor_model_def, critic_def=critic_model_def,
				  seed=None ) :

		self.gamma = gamma
		self.tau = tau
		self.minibatch_size = minibatch_size

		# Definition of the target entropy:
		if target_entropy is None :
			target_entropy = -a_dim
		elif target_entropy > 0 :
			raise ValueError( 'Wrong argument for the target entropy: H has to be negative' )
		self.H = target_entropy

		# Definition of each learning rate:
		if actor_lr is None :
			actor_lr = learning_rate
		if critic_lr is None :
			critic_lr = learning_rate
		if alpha_lr is None :
			alpha_lr = learning_rate

		# Temperature variable before it is constrained to be positive:
		self._alpha_unconstrained = tf.Variable( np.log( np.exp( alpha0 ) - 1 ), dtype=tf.float32 )

		# Number of iterations done:
		self.n_iter = 0

		# Instantiation of the replay buffer:
		self.replay_buffer = deque( maxlen=int( buffer_size ) )

		# Set the different random seeds:
		random.seed( seed )
		tf.random.set_seed( seed )


		# Declaration of the scaling factors:
		if state_scale is not None :
			self.state_scale = tf.constant( state_scale, tf.float32, name='state_scale' )
		else :
			self.state_scale = None

		if action_scale is not None :
			self.action_scale = tf.constant( action_scale, tf.float32, name='action_scale' )
		else :
			self.action_scale = None


		# Instantiation of the actor network:
		self.actor = actor_def( s_dim, a_dim )
		self.actor_optimizer = tf.optimizers.Adam( learning_rate=actor_lr )


		# Instantiation of the critic networks:
		self.critics = []
		for _ in range( 2 ) :
			critic = {}
			# Instantiation of the Q-function networks:
			critic['network'] = critic_def( s_dim, a_dim )
			critic['optimizer'] = tf.optimizers.Adam( learning_rate=critic_lr )
			# Instantiation of the target Q-function networks:
			critic['target_network'] = critic_def( s_dim, a_dim )
			# Synchronization of the target network parameters:
			for target_params, params in zip( critic['target_network'].trainable_variables, critic['network'].trainable_variables ) :
				target_params.assign( params )
			self.critics.append( critic )


		# Instantiation of the temperature optimizer:
		self.alpha_optimizer = tf.optimizers.Adam( learning_rate=alpha_lr )


	@tf.function
	def _infer_Q_values( self, critic_model, states, actions, return_reg=False ) :
		states = tf.cast( states, tf.float32 )
		actions = tf.cast( actions, tf.float32 )

		if self.state_scale is not None :
			states = tf.divide( states, self.state_scale, 'scale_states' )

		if self.action_scale is not None :
			actions = tf.divide( actions, action_scale, 'scale_actions' )

		# Inference from the critic network:
		Q_values = critic_model( [ states, actions ] )

		if return_reg :
			# Return the critic network regularization beside the Q-values:
			return Q_values, tf.reduce_sum( critic_model.losses ) if critic_model.losses else tf.zeros( 1 )
		return Q_values


	@tf.function
	def _infer_actions( self, states, sample=False, return_reg=False ) :
		states = tf.cast( states, tf.float32 )

		if self.state_scale is not None :
			states = tf.divide( states, self.state_scale, 'scale_states' )

		# Inference from the actor network:
		mu, sigma = self.actor( states )

		if sample :
			u = tf.random.normal( mu.shape, mu, sigma )
		else :
			u = mu

		# Squashing of the actions:
		actions = tf.tanh( u )

		if self.action_scale is not None :
			actions = tf.multiply( actions, action_scale, 'scale_actions' )

		a_dict = { 'a': actions, 'u': u, 'mu': mu, 'sigma': sigma }
		if return_reg :
			# Addition of the actor network regularization to the outputs:
			a_dict['reg'] = tf.reduce_sum( self.actor.losses ) if self.actor.losses else tf.zeros( 1 )

		return a_dict


	@tf.function
	def _get_actions_and_log_pis( self, states, sample, return_reg=False ) :

		a_dict = self._infer_actions( states, sample, return_reg )

		# Unbounded Gaussian action distributions:
		u_distribs = tfp.distributions.Normal( a_dict['mu'], a_dict['sigma'], allow_nan_stats=False )
		# Log-likelihood of the policy taking the squashing function into account:
		log_pis = u_distribs.log_prob( a_dict['u'] ) - tf.reduce_sum( tf.math.log( 1 - tf.tanh( a_dict['u'] )**2 + 1e-6 ), axis=1, keepdims=True )

		if return_reg :
			return a_dict['a'], log_pis, a_dict['reg']
		return a_dict['a'], log_pis


	@tf.function
	def _train_Q_networks( self, batch ) :

		states = batch['states']
		actions = batch['actions']
		rewards = tf.cast( batch['rewards'], tf.float32 )
		masks = tf.cast( tf.logical_not( batch['terminals'] ), tf.float32 )
		next_states = batch['next_states']

		next_actions, next_log_pis = self._get_actions_and_log_pis( next_states, sample=False )

		next_Q_values_list = [ self._infer_Q_values( critic['target_network'], next_states, next_actions ) for critic in self.critics ]
		next_Q_values = tf.reduce_min( next_Q_values_list, axis=0 )

		# Computation of the soft temporal difference:
		Q_targets = rewards + self.gamma*( next_Q_values - self.alpha()*next_log_pis )*masks

		critic_losses = []
		for critic in self.critics :
			with tf.GradientTape() as tape :

				Q_values, reg_loss = self._infer_Q_values( critic['network'], states, actions, return_reg=True )

				# Minimization of the soft Bellman residual:
				critic_loss = 0.5*tf.reduce_mean( tf.losses.mean_squared_error( Q_targets, Q_values ) )
				# Add the regularization from the critic network:
				critic_loss += reg_loss

			gradients = tape.gradient( critic_loss, critic['network'].trainable_variables )
			critic['optimizer'].apply_gradients( zip( gradients, critic['network'].trainable_variables ) )

			critic_losses.append( critic_loss )

		return tf.reduce_mean( critic_losses )


	@tf.function
	def _train_actor_network( self, states ) :

		with tf.GradientTape() as tape :

			actions, log_pis, reg_loss = self._get_actions_and_log_pis( states, sample=True, return_reg=True )

			Q_values_list = [ self._infer_Q_values( critic['network'], states, actions ) for critic in self.critics ]
			Q_values = tf.reduce_min( Q_values_list, axis=0 )

			# Minimization of the KL-divergence from the policy to the exponential of the soft Q-function:
			actor_loss = tf.reduce_mean( self.alpha()*log_pis - Q_values )
			# Add the regularization from the actor network:
			actor_loss += reg_loss

		gradients = tape.gradient( actor_loss, self.actor.trainable_variables )
		self.actor_optimizer.apply_gradients( zip( gradients, self.actor.trainable_variables ) )

		return actor_loss


	@tf.function
	def alpha( self ) :
		""" Return the positive-only entropy temperature """
		return tf.math.softplus( self._alpha_unconstrained )


	@tf.function
	def _update_temperature( self, states ) :

		actions, log_pis = self._get_actions_and_log_pis( states, sample=False )

		with tf.GradientTape() as tape :

			# Constraint of the average entropy of the policy to a desired minimum value:
			alpha_loss = -self.alpha()*tf.reduce_mean( log_pis + self.H )

		gradients = tape.gradient( alpha_loss, [ self._alpha_unconstrained ] )
		self.alpha_optimizer.apply_gradients( zip( gradients, [ self._alpha_unconstrained ] ) )

		return alpha_loss


	@tf.function
	def _update_target_Q_networks( self ) :
		for critic in self.critics :
			# Tracking of the Q-function networks by the target networks with an exponentially moving average of the weights:
			for target_params, params in zip( critic['target_network'].trainable_variables, critic['network'].trainable_variables ) :
				target_params.assign( self.tau*params + ( 1 - self.tau )*target_params )


	def _sample_batch( self, batch_size ) :

		raw_batch = random.sample( self.replay_buffer, min( len( self.replay_buffer ), batch_size ) )

		batch = {}
		batch['states']      = np.array( [ _[0] for _ in raw_batch ] )
		batch['actions']     = np.array( [ _[1] if np.shape( _[1] ) else [ _[1] ] for _ in raw_batch ] )
		batch['rewards']     = np.array( [ [ _[2] ] for _ in raw_batch ] )
		batch['terminals']   = np.array( [ [ _[3] ] for _ in raw_batch ] )
		batch['next_states'] = np.array( [ _[4] for _ in raw_batch ] )

		return batch


	def train( self, iterations=1 ) :

		if len( self.replay_buffer ) < self.minibatch_size :
			return 0

		Lt = 0

		for _ in trange( iterations, desc='Training the networks', leave=False ) :

			self.n_iter += 1

			# Randomly pick samples in the replay buffer:
			batch = self._sample_batch( self.minibatch_size )

			Lt += self._train_Q_networks( batch )

			self._train_actor_network( batch['states'] )

			self._update_temperature( batch['states'] )

			self._update_target_Q_networks()

		return float( Lt )/iterations


	def stoch_action( self, s ) :
		if s.ndim < 2 : s = s[np.newaxis, :]

		a_dict = self._infer_actions( s, sample=True )

		return tf.squeeze( a_dict['a'] ).numpy()


	def best_action( self, s, return_stddev=False ) :
		if s.ndim < 2 : s = s[np.newaxis, :]

		a_dict = self._infer_actions( s )

		if return_stddev :
			return tf.squeeze( a_dict['a'] ).numpy(), tf.squeeze( a_dict['sigma'] ).numpy()
		return tf.squeeze( a_dict['a'] ).numpy()


	def get_Q_value( self, s, a ) :
		if s.ndim < 2 : s = s[np.newaxis, :]
		if isinstance( a, np.ndarray ) and a.ndim < 2 : a = a[np.newaxis, :]

		Q_value_list = [ self._infer_Q_values( critic['network'], s, a ) for critic in self.critics ]
		Q_value = tf.reduce_min( Q_value_list, axis=0 )

		return tf.squeeze( Q_value ).numpy()


	def get_V_value( self, s ) :
		if s.ndim < 2 : s = s[np.newaxis, :]

		a_dict = self._infer_actions( s )
		V_value = self.get_Q_value( s, a_dict['a'] )

		return tf.squeeze( V_value ).numpy()

	
	def _save_optimizer_weights( self, optimizer, filename ) :
		np.save( filename, optimizer.get_weights(), allow_pickle=True )

	
	def _load_optimizer_weights( self, optimizer, model, filename ) :
		weights = np.load( filename + '.npy', allow_pickle=True )
		optimizer._create_all_weights( model.trainable_variables )
		optimizer.set_weights( weights )


	def save( self, directory, hdf5=True, optimizer_weights=True ) :

		extension = '.hdf5' if hdf5 else ''

		self.actor.save( directory + '/actor' + extension )
		if optimizer_weights :
			self._save_optimizer_weights( self.actor_optimizer, directory + '/actor_optimizer_weights' )

		for i, critic in enumerate( self.critics ) :
			critic['network'].save( directory + '/critic_%i%s' % ( i + 1, extension ) )
			critic['target_network'].save( directory + '/critic_%i_target%s' % ( i + 1, extension ) )
			if optimizer_weights :
				self._save_optimizer_weights( critic['optimizer'], directory + '/critic_%i_optimizer_weights' % ( i + 1 ) )


	def load( self, directory, hdf5=True, optimizer_weights=True ) :

		extension = '.hdf5' if hdf5 else ''

		self.actor = keras.models.load_model( directory + '/actor' + extension, compile=False )
		if optimizer_weights :
			self._load_optimizer_weights( self.actor_optimizer, self.actor, directory + '/actor_optimizer_weights' )

		for i, critic in enumerate( self.critics ) :
			critic['network'] = keras.models.load_model( directory + '/critic_%i%s' % ( i + 1, extension ), compile=False )
			critic['target_network'] = keras.models.load_model( directory + '/critic_%i_target%s' % ( i + 1, extension ), compile=False )
			if optimizer_weights :
				self._load_optimizer_weights( critic['optimizer'], critic['network'], directory + '/critic_%i_optimizer_weights' % ( i + 1 ) )


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
