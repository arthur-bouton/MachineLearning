""" 
Implementation of the Twin Delayed Deep Deterministic policy gradient (TD3) algorithm with automated entropy temperature adjustment [1] for continuous-state and continuous-action spaces using TensorFlow 2.

[1] Fujimoto, Scott, Herke Van Hoof, and David Meger. "Addressing function approximation error in actor-critic methods." arXiv preprint arXiv:1802.09477 (2018).

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
import pickle
import yaml


from tensorflow import keras
from tensorflow.keras import layers


# Default actor network:
def actor_model_def( s_dim, a_dim ) :

	states = keras.Input( shape=(s_dim,) )

	x = layers.Dense( 400, activation='relu' )( states )
	x = layers.Dense( 300, activation='relu' )( x )

	actions = layers.Dense( a_dim, activation='tanh' )( x )

	return keras.Model( states, actions )


# Default critic network:
def critic_model_def( s_dim, a_dim ) :

	states  = keras.Input( shape=(s_dim,) )
	actions = keras.Input( shape=(a_dim,) )

	x = layers.Concatenate()( [ states, actions ] )
	x = layers.Dense( 400, activation='relu' )( x )
	x = layers.Concatenate()( [ x, actions ] )
	x = layers.Dense( 300, activation='relu' )( x )
	Q_value = layers.Dense( 1, activation='linear' )( x )

	return keras.Model( [ states, actions ], Q_value )


class TD3 :
	"""
	Twin Delayed Deep Deterministic policy gradient algorithm.

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
	tau : float, optional, default: 5e-3
		Soft target update factor.
	policy_update_delay : int, optional, default: 2
		Number of critic updates for one policy update.
	policy_reg_sigma : float, optional, default: 0.2
		Standard deviation of the target policy regularization noise.
	policy_reg_bound : float, optional, default: 0.5
		Bounds of the target policy regularization noise.
	buffer_size : int, optional, default: 1e6
		Maximal size of the replay buffer.
	minibatch_size : int, optional, default: 256
		Size of each minibatch.
	learning_rate : float, optional, default: 3e-4
		Default learning rate used for all the networks.
	actor_lr : float, optional, default: None
		Learning rate to use for the optimization of the actor network.
		If None, learning_rate is used.
	critic_lr : float, optional, default: None
		Learning rate to use for the optimization of the critic networks.
		If None, learning_rate is used.
	actor_def : function, optional, default: actor_model_def
		Function defining the actor model.
		It has to take the dimension of the state and the action spaces
		as inputs and return a Keras model.
		If needed, the action squashing has to be included in the model.
	critic_def : function, optional, default: critic_model_def
		Function defining the critic model.
		It has to take the dimension of the state and the action spaces
		as inputs and return a Keras model.
	seed : int, optional, default: None
		Random seed for the initialization of random generators.

	Examples
	--------
	# Fill the replay buffer with transitions:
	td3.replay_buffer.append(( state, action, reward, is_terminal, next_state ))

	# Train the networks:
	loss = td3.train( nb_iterations )

	# Infer the best actions from the current policy:
	action = td3.get_action( state )

	"""

	def __init__( self, s_dim, a_dim, state_scale=None, action_scale=None,
	              gamma=0.99, target_entropy=None, tau=5e-3, buffer_size=1e6, minibatch_size=100,
				  learning_rate=1e-3, actor_lr=None, critic_lr=None,
				  policy_update_delay=2, policy_reg_sigma=0.2, policy_reg_bound=0.5,
				  actor_def=actor_model_def, critic_def=critic_model_def,
				  seed=None ) :

		self._variables = {}
		self._variables['gamma'] = gamma
		self._variables['tau'] = tau
		self._variables['policy_update_delay'] = policy_update_delay
		self._variables['policy_reg_sigma'] = policy_reg_sigma
		self._variables['policy_reg_bound'] = policy_reg_bound
		self._variables['minibatch_size'] = minibatch_size

		# Define each learning rate:
		if actor_lr is None :
			actor_lr = learning_rate
		if critic_lr is None :
			critic_lr = learning_rate
		self._variables['actor_lr'] = actor_lr
		self._variables['critic_lr'] = critic_lr

		# Number of iterations done:
		self._variables['n_iter'] = 0

		# Instantiate the replay buffer:
		self.replay_buffer = deque( maxlen=int( buffer_size ) )

		# Set the different random seeds:
		random.seed( seed )
		tf.random.set_seed( seed )


		# Define the scaling factors:
		self._variables['state_scale'] = state_scale
		self._variables['action_scale'] = action_scale


		# Instantiate the actor network:
		self.actor = actor_def( s_dim, a_dim )
		self.actor_optimizer = tf.optimizers.Adam( learning_rate=actor_lr )
		# Instantiate the actor target network:
		self.actor_target_network = actor_def( s_dim, a_dim )
		# Synchronize the target network parameters:
		for target_params, params in zip( self.actor_target_network.trainable_variables, self.actor.trainable_variables ) :
			target_params.assign( params )


		# Instantiate the critic networks:
		self.critics = []
		for _ in range( 2 ) :
			critic = {}
			# Instantiate the Q-function networks:
			critic['network'] = critic_def( s_dim, a_dim )
			critic['optimizer'] = tf.optimizers.Adam( learning_rate=critic_lr )
			# Instantiate the target Q-function networks:
			critic['target_network'] = critic_def( s_dim, a_dim )
			# Synchronize the target network parameters:
			for target_params, params in zip( critic['target_network'].trainable_variables, critic['network'].trainable_variables ) :
				target_params.assign( params )
			self.critics.append( critic )


	@tf.function
	def _infer_Q_values( self, critic_model, states, actions, return_reg=False, training=False ) :
		states = tf.cast( states, tf.float32 )
		actions = tf.cast( actions, tf.float32 )

		if self._variables['state_scale'] is not None :
			states /= self._variables['state_scale']

		if self._variables['action_scale'] is not None :
			actions /= self._variables['action_scale']

		# Inference from the critic network:
		Q_values = critic_model( [ states, actions ], training=training )

		if return_reg :
			# Return the critic network regularization beside the Q-values:
			return Q_values, tf.reduce_sum( critic_model.losses ) if critic_model.losses else tf.zeros( 1 )
		return Q_values


	@tf.function
	def _infer_actions( self, states, return_reg=False, training=False, target=False ) :
		states = tf.cast( states, tf.float32 )

		if self._variables['state_scale'] is not None :
			states /= self._variables['state_scale']

		if target :
			# Inference from the actor target network:
			actions = self.actor_target_network( states )
		else :
			# Inference from the actor network:
			actions = self.actor( states, training=training )

		if self._variables['action_scale'] is not None :
			actions *= self._variables['action_scale']

		if return_reg :
			# Return the actor network regularization beside the actions:
			return actions, tf.reduce_sum( self.actor.losses ) if self.actor.losses else tf.zeros( 1 )
		return actions


	@tf.function
	def _train_Q_networks( self, batch ) :

		states = batch['states']
		actions = batch['actions']
		rewards = tf.cast( batch['rewards'], tf.float32 )
		masks = tf.cast( tf.logical_not( batch['terminals'] ), tf.float32 )
		next_states = batch['next_states']

		next_actions = self._infer_actions( next_states, target=True )

		# Add the target policy smoothing regularization:
		policy_noise = tf.random.normal( next_actions.shape, 0, self._variables['policy_reg_sigma'] )
		policy_noise = tf.clip_by_value( policy_noise, -self._variables['policy_reg_bound'], self._variables['policy_reg_bound'] )
		if self._variables['action_scale'] is not None :
			policy_noise *= self._variables['action_scale']
		next_actions += policy_noise

		# Clipped double Q-learning:
		next_Q_values_list = [ self._infer_Q_values( critic['target_network'], next_states, next_actions ) for critic in self.critics ]
		next_Q_values = tf.reduce_min( next_Q_values_list, axis=0 )

		# Compute the expected return:
		Q_targets = rewards + self._variables['gamma']*next_Q_values*masks

		critic_losses = []
		for critic in self.critics :
			with tf.GradientTape() as tape :

				Q_values, reg_loss = self._infer_Q_values( critic['network'], states, actions, return_reg=True, training=True )

				# Minimize the Bellman residual:
				critic_loss = tf.reduce_mean( tf.losses.MeanSquaredError()( Q_targets, Q_values ) )
				# Add the regularization from the critic network:
				critic_loss += reg_loss

			gradients = tape.gradient( critic_loss, critic['network'].trainable_variables )
			critic['optimizer'].apply_gradients( zip( gradients, critic['network'].trainable_variables ) )

			critic_losses.append( critic_loss )

		return tf.reduce_mean( critic_losses )


	@tf.function
	def _train_actor_network( self, states ) :

		with tf.GradientTape() as tape :

			actions, reg_loss = self._infer_actions( states, return_reg=True, training=True )

			# Clipped double Q-learning:
			Q_values_list = [ self._infer_Q_values( critic['network'], states, actions ) for critic in self.critics ]
			Q_values = tf.reduce_min( Q_values_list, axis=0 )

			# Update the actor so as to maximize the Q-value (deterministic policy gradient):
			actor_loss = tf.reduce_mean( -Q_values )
			# Add the regularization from the actor network:
			actor_loss += reg_loss

		gradients = tape.gradient( actor_loss, self.actor.trainable_variables )
		self.actor_optimizer.apply_gradients( zip( gradients, self.actor.trainable_variables ) )

		return actor_loss


	@tf.function
	def _update_target_networks( self ) :
		# Tracking of the actor and Q-function networks by the target networks with an exponentially moving average of the weights:
		for target_params, params in zip( self.actor_target_network.trainable_variables, self.actor.trainable_variables ) :
			target_params.assign( self._variables['tau']*params + ( 1 - self._variables['tau'] )*target_params )
		for critic in self.critics :
			for target_params, params in zip( critic['target_network'].trainable_variables, critic['network'].trainable_variables ) :
				target_params.assign( self._variables['tau']*params + ( 1 - self._variables['tau'] )*target_params )


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

		if len( self.replay_buffer ) < self._variables['minibatch_size'] :
			return 0

		Q_loss = 0

		for _ in trange( iterations, desc='Training the networks', leave=False ) :

			self._variables['n_iter'] += 1

			for _ in range( self._variables['policy_update_delay'] ) :

				# Randomly pick samples in the replay buffer:
				batch = self._sample_batch( self._variables['minibatch_size'] )

				Q_loss += self._train_Q_networks( batch )

			# Delayed policy updates:
			self._train_actor_network( batch['states'] )

			self._update_target_networks()

		return float( Q_loss )/( iterations*self._variables['policy_update_delay'] )


	@property
	def n_iter( self ) :
		return self._variables['n_iter']


	def get_action( self, s ) :
		if s.ndim < 2 : s = s[np.newaxis, :]

		actions = self._infer_actions( s )

		return tf.squeeze( actions ).numpy()


	def get_Q_value( self, s, a ) :
		if s.ndim < 2 : s = s[np.newaxis, :]
		if isinstance( a, np.ndarray ) and a.ndim < 2 : a = a[np.newaxis, :]

		# Clipped double Q-learning:
		Q_value_list = [ self._infer_Q_values( critic['network'], s, a ) for critic in self.critics ]
		Q_value = tf.reduce_min( Q_value_list, axis=0 )

		return tf.squeeze( Q_value ).numpy()


	def get_V_value( self, s ) :
		if s.ndim < 2 : s = s[np.newaxis, :]

		actions = self._infer_actions( s )
		V_value = self.get_Q_value( s, actions )

		return tf.squeeze( V_value ).numpy()

	
	def _save_optimizer( self, optimizer, filename ) :
		with open( filename + '.pkl', 'wb' ) as f :
			pickle.dump( optimizer, f )


	def _load_optimizer( self, filename ) :
		with open( filename + '.pkl', 'rb' ) as f :
			return pickle.load( f )


	def save( self, directory, extension='keras', optimizers=True ) :

		# Save the actor model, its optimizer and its target network:
		self.actor.save( directory + '/actor.' + extension )
		if optimizers :
			self._save_optimizer( self.actor_optimizer, directory + '/actor_optimizer' )
		self.actor_target_network.save( directory + '/actor_target.' + extension )

		# Save the critic models and their optimizers:
		for i, critic in enumerate( self.critics ) :
			critic['network'].save( f'{directory}/critic_{i + 1}.{extension}' )
			critic['target_network'].save( f'{directory}/critic_{i + 1}_target.{extension}' )
			if optimizers :
				self._save_optimizer( critic['optimizer'], f'{directory}/critic_{i + 1}_optimizer' )

		# Save the internal variables:
		with open( directory + '/variables.yaml', 'w' ) as f :
			f.write( '# TD3 variables:\n' )
			yaml.dump( self._variables, f )


	def load( self, directory, extension='keras', optimizers=True ) :

		# Load the actor model, its optimizer and its target network:
		self.actor = keras.models.load_model( directory + '/actor.' + extension, compile=False )
		if optimizers :
			self.actor_optimizer = self._load_optimizer( directory + '/actor_optimizer' )
		self.actor_target_network = keras.models.load_model( directory + '/actor_target.' + extension, compile=False )

		# Load the critic models and their optimizers:
		for i, critic in enumerate( self.critics ) :
			critic['network'] = keras.models.load_model( f'{directory}/critic_{i + 1}.{extension}', compile=False )
			critic['target_network'] = keras.models.load_model( f'{directory}/critic_{i + 1}_target.{extension}', compile=False )
			if optimizers :
				critic['optimizer'] = self._load_optimizer( f'{directory}/critic_{i + 1}_optimizer' )

		# Load the internal variables:
		with open( directory + '/variables.yaml', 'r' ) as f :
			self._variables = yaml.load( f, Loader=yaml.FullLoader )

		# Update the optimizers' learning rates:
		self.actor_optimizer.learning_rate = self._variables['actor_lr']
		for ciritc in self.critics :
			critic['optimizer'].learning_rate = self._variables['critic_lr']


	def save_replay_buffer( self, filename ) :
		with open( filename, 'wb' ) as f :
			pickle.dump( self.replay_buffer, f )


	def load_replay_buffer( self, filename ) :
		try :
			with open( filename, 'rb' ) as f :
				temp_buf = pickle.load( f )
			self.replay_buffer = temp_buf
			return True
		except IOError :
			return False
