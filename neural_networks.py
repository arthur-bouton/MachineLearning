""" 
Implementation of scalable Multilayer Perceptron (MLP) and Radial Basis Function network (RBF) with numpy only.

Author: Arthur Bouton [arthur.bouton@gadz.org]
"""
import numpy as np
from copy import deepcopy

#import warnings
#warnings.filterwarnings( 'error' )

#np.seterr( all='raise' )


class MLP :
	""" Multilayer Perceptron """

	def __init__( self, layers, lambda0=0.01, momentum=0, adaptive_stepsize=True, activation_function='sigmoid', seed=None ) :
		
		self._layers = layers
		self._alpha = lambda0
		self.mu = momentum
		self.adaptive_stepsize = adaptive_stepsize

		self._Nl = len( self._layers )

		# Ranges for initialization of weights and bias:
		w0_range = [ -1.0, 1.0 ]
		b0_range = [ -0.5, 0.5 ]

		# Initialization of weights and bias:
		np.random.seed( seed )
		self._W = []
		self._B = []
		for i in range( self._Nl - 1 ) :
			self._W.append( np.random.uniform( w0_range[0], w0_range[1], ( self._layers[i], self._layers[i+1] ) ) )
			self._B.append( np.random.uniform( b0_range[0], b0_range[1], self._layers[i+1] ) )

		# Activation functions and their derivatives:
		self._F = []
		self._dF = []
		if activation_function == 'sigmoid' :
			for i in range( self._Nl - 2 ) :
				self._F.append( lambda x : 1/( 1 + np.exp( -x ) ) )
				self._dF.append( lambda x : np.exp( -x )/( 1 + np.exp( -x ) )**2 )
				#self._F.append( lambda x : 1/( 1 + np.exp( -np.clip( x, -500, 500 ) ) ) )
				#self._dF.append( lambda x : np.exp( -np.clip( x, -500, 500 ) )/( 1 + np.exp( -np.clip( x, -500, 500 ) ) )**2 )
		elif activation_function == 'relu' :
			for i in range( self._Nl - 2 ) :
				self._F.append( lambda x : np.where( x > 0, x, 0 ) )
				self._dF.append( lambda x : np.where( x > 0, 1, 0 ) )
		elif activation_function == 'gaussian' :
			for i in range( self._Nl - 2 ) :
				self._F.append( lambda x : np.exp( -x**2 ) )
				self._dF.append( lambda x : -2*x*np.exp( -x**2 ) )
		else :
			raise ValueError( '%s is not known' % activation_function )

		# Linear activation for the final layer:
		self._F.append( lambda x : x )
		self._dF.append( lambda x : np.ones( x.shape ) )

		self._Qbatch = 0.
		self._nbatch = 0
		self._Qbak = []
		self._alphabak = []

		self._dWprev = []
		self._dBprev = []
		for i in range( self._Nl - 1 ) :
			self._dWprev.append( np.zeros( ( self._layers[i], self._layers[i+1] ) ) )
			self._dBprev.append( np.zeros( self._layers[i+1] ) )

		self._Wbak = []
		self._Bbak = []
	
	def _backprop( self, inputs, targets ) :

		# Propagation:
		V = []
		O = [ inputs ]
		for i in range( self._Nl - 1 ) :
			V.append( np.dot( O[i], self._W[i] ) + self._B[i] )
			O.append( self._F[i]( V[i] ) )

		# Backpropagation:
		D = [ np.multiply( O[-1] - targets, self._dF[-1]( V[-1] ) ) ]
		for i in range( self._Nl - 2, 0, -1 ) :
			D.insert( 0, np.multiply( self._dF[i-1]( V[i-1] ), np.dot( self._W[i], D[0] ) ) )

		# Computation of the updates:
		dW = []
		dB = []
		for i in range( self._Nl - 1 ) :
			dW.append( self._alpha*np.outer( O[i], D[i] ) )
			dB.append( self._alpha*D[i] )

		# Computation of the cost function:
		self._Qbatch += np.dot( O[-1] - targets, O[-1] - targets )/2
		self._nbatch += 1

		return dW, dB

	def _update( self, dW, dB ) :

		for i in range( self._Nl - 1 ) :
			self._W[i] -= dW[i] + self.mu*self._dWprev[i]
			self._B[i] -= dB[i] + self.mu*self._dBprev[i]
			self._dWprev[i] = dW[i]
			self._dBprev[i] = dB[i]
	
	def inc_training( self, inputs, targets ) :

		# Computation of the updates:
		dW, dB = self._backprop( inputs, targets )

		# Updates:
		self._update( dW, dB )

	def end_of_batch( self ) :

		if self._nbatch > 0 :

			# Computation of the avarage cost:
			self._Qbak.append( self._Qbatch/self._nbatch )
			self._Qbatch = 0.
			self._nbatch = 0
			self._alphabak.append( self._alpha )

			self._Wbak.append( deepcopy( self._W ) )
			self._Bbak.append( deepcopy( self._B ) )

			# Step size adaptation:
			if self.adaptive_stepsize and len( self._Qbak ) > 1 :
				if self._Qbak[-1] - self._Qbak[-2] <= 0 :
					self._alpha *= 1.1
				else :
					self._alpha *= 0.5

			return self._Qbak[-1], self._alpha

		else :
			return 0, 0

	def batch_training( self, data_inputs, data_targets ) :

		if len( data_inputs ) > 0 :

			# Computation of the avarage updates:
			dW_total, dB_total = self._backprop( data_inputs[0], data_targets[0] )
			for i in range( 1, len( data_inputs ) ) :
				dW, dB = self._backprop( data_inputs[i], data_targets[i] )
				for i in range( self._Nl - 1 ) :
					dW_total[i] += dW[i]
					dB_total[i] += dB[i]
			for i in range( self._Nl - 1 ) :
				dW_total[i] = dW_total[i]/self._nbatch
				dB_total[i] = dB_total[i]/self._nbatch

			# Updates:
			self._update( dW_total, dB_total )

			return self.end_of_batch()

		else :
			return 0, 0

	def eval( self, inputs ) :

		# Propagation:
		O = inputs
		for i in range( self._Nl - 1 ) :
			O = self._F[i]( np.dot( O, self._W[i] ) + self._B[i] )

		return O

	def save( self, filename='mlp_wb' ) :

		np.save( filename, [ self._W, self._B ] )

	def load( self, filename='mlp_wb' ) :

		data = np.load( filename + '.npy' )
		self._W = data[0]
		self._B = data[1]

	def plot( self, pause=True, weights=True, lambda0=False, name='' ) :
		from matplotlib.pyplot import figure, plot, yscale, show, subplots

		figQ = figure( 'Cost function ' + name )
		plot( self._Qbak )
		yscale( 'log' )

		if lambda0 :
			figAlpha = figure( 'Lambda ' + name )
			plot( self._alphabak )

		if weights :
			figW, axw = subplots( self._Nl - 1, sharex=True )
			figW.canvas.set_window_title( 'Weights ' + name )
			figB, axb = subplots( self._Nl - 1, sharex=True )
			figB.canvas.set_window_title( 'Biais ' + name )
			for l in range( self._Nl - 1 ) :
				for n in range( self._layers[l] ) :
					for w in range( self._layers[l+1] ) :
						axw[l].plot( [ W[l][n][w] for W in self._Wbak ] )
					axb[l].plot( [ B[l][w] for B in self._Bbak ] )

		if ( pause ) :
			show()


class RBF :
	""" Radial Basis Function network """

	def __init__( self, dim, distrib, sigma, initial_value=0, lambda0=0.01, adaptive_stepsize=True, normalized=True ) :
		
		N_distr = len( distrib )
		self._Nk = N_distr**dim
		self._sigma2 = sigma**2
		self._alpha = lambda0
		self.adaptive_stepsize = adaptive_stepsize
		self._normalize = normalized

		# Definition of the kernel centers:
		self._centers = []
		for k in range( self._Nk ) :
			C = [ distrib[ k%N_distr ] ]
			for i in range( 1, dim ) :
				C.append( distrib[ ( k//( N_distr**i ) )%N_distr ] )
			self._centers.append( np.array( C ) )

		# Initialization of the weights:
		self._W = [ initial_value ]*self._Nk

		self._Qbatch = 0.
		self._nbatch = 0
		self._Qbak = []
		self._alphabak = []

	def _compute_updates( self, inputs, targets ) :

		S = 0.
		ek = []
		et = 0.
		for k in range( self._Nk ) :
			v = inputs - self._centers[k]
			ek.append( np.exp( -np.dot( v, v )/self._sigma2 ) )
			S += self._W[k]*ek[k]
			et += ek[k]

		# Normalization:
		if self._normalize :
		#if self._normalize and et > 1e-7 :
			S /= et
			#with np.errstate( all='raise' ) :
				#try :
					#S /= et
				#except FloatingPointError :
					#S = 0

		# Computation of the error:
		err = targets - S

		# Computation of the updates:
		dW = []
		for k in range( self._Nk ) :
			dW.append( self._alpha*err*ek[k]/( et if self._normalize else 1 ) )

		# Computation of the cost function:
		self._Qbatch += np.dot( err, err )/2
		self._nbatch += 1
		
		return dW
	
	def inc_training( self, inputs, targets ) :

		# Computation of the updates:
		dW = self._compute_updates( inputs, targets )

		# Updates:
		for k in range( self._Nk ) :
			self._W[k] += dW[k]

	def end_of_batch( self ) :

		if self._nbatch > 0 :

			# Computation of the avarage cost:
			self._Qbak.append( self._Qbatch/self._nbatch )
			self._Qbatch = 0.
			self._nbatch = 0
			self._alphabak.append( self._alpha )

			# Step size adaptation:
			if self.adaptive_stepsize and len( self._Qbak ) > 1 :
				if self._Qbak[-1] - self._Qbak[-2] <= 0 :
					self._alpha *= 1.1
				else :
					self._alpha *= 0.5

			return self._Qbak[-1], self._alpha

		else :
			return 0, 0

	def batch_training( self, data_inputs, data_targets ) :

		if len( data_inputs ) > 0 :

			# Computation of the avarage updates:
			dW_total = self._compute_updates( data_inputs[0], data_targets[0] )
			for i in range( 1, len( data_inputs ) ) :
				dW = self._compute_updates( data_inputs[i], data_targets[i] )
				for k in range( self._Nk ) :
					dW_total[k] += dW[k]

			# Updates:
			for k in range( self._Nk ) :
				self._W[k] += dW_total[k]/self._nbatch

			return self.end_of_batch()

		else :
			return 0, 0

	def eval( self, inputs ) :

		S = 0.
		ek = []
		et = 0.
		for k in range( self._Nk ) :
			v = inputs - self._centers[k]
			ek.append( np.exp( -np.dot( v, v )/self._sigma2 ) )
			S += self._W[k]*ek[k]
			et += ek[k]

		# Normalization:
		if self._normalize :
			S /= et
			#with np.errstate( all='raise' ) :
				#try :
					#S /= et
				#except FloatingPointError :
					#S = 0

		return S

	def save( self, filename='rbf_wb' ) :

		np.save( filename, self._W )

	def load( self, filename='rbf_wb' ) :

		self._W = np.load( filename + '.npy' )

	def plot( self, pause=True, lambda0=False, name='' ) :
		from matplotlib.pyplot import figure, plot, yscale, show, subplots

		figQ = figure( 'Cost function ' + name )
		plot( self._Qbak )
		yscale( 'log' )

		if lambda0 :
			figAlpha = figure( 'Lambda ' + name )
			plot( self._alphabak )

		if ( pause ) :
			show()


