import signal


class Loop_handler :
	""" 
	Context manager which allows the SIGINT signal to be processed asynchronously.

	Author: Arthur Bouton [arthur.bouton@gadz.org]
	"""

	def __init__( self ) :
		self._end = False
	
	def check_interruption( self ) :
		return self._end

	def __enter__( self ) :

		def handler( signum, frame ) :
			self._end = True

		self._original_handler = signal.getsignal( signal.SIGINT )
		signal.signal( signal.SIGINT, handler )

		return self.check_interruption

	def __exit__( self, type, value, traceback ) :
		signal.signal( signal.SIGINT, self._original_handler )



import matplotlib.pyplot as plt
import collections
import atexit
import sys


class Monitor :
	""" 
	Plot variables incrementally.

	Author: Arthur Bouton [arthur.bouton@gadz.org]
	"""

	def __init__( self, n_var=1, labels=None, titles=None, xlabel=None, name=None, log=False, keep=True, x_step=1, plot_kwargs=None ) :

		if not isinstance( n_var, collections.Iterable ) : n_var = [ n_var ]

		# Create the figure:
		self.fig, self.axes = plt.subplots( len( n_var ), sharex=True )

		if not isinstance( self.axes, collections.Iterable ) : self.axes = [ self.axes ]

		# Name the figure window:
		if name is not None :
			self.fig.canvas.set_window_title( name )
		else :
			self.fig.canvas.set_window_title( sys.argv[0] )

		# Initialize the subplot(s):
		self._xdata = []
		self._ydata = []
		self._lines = []

		if isinstance( plot_kwargs, dict ) :
			kwargs = plot_kwargs
		else :
			kwargs = {}

		for i, ax in enumerate( self.axes ) :
			if n_var[i] < 1 :
				raise ValueError( 'Wrong argument n_var: there cannot be less than one variable on a subplot' )
			for j in range( n_var[i] ) :

				self._ydata.append( [] )

				if isinstance( plot_kwargs, list ) :
					if len( plot_kwargs ) > sum( n_var[:i] ) + j :
						kwargs = plot_kwargs[sum( n_var[:i] ) + j]
					else :
						kwargs = {}

				self._lines.append( ax.plot( [], **kwargs )[0] )

			ax.grid( True )
		if xlabel is not None :
			ax.set_xlabel( xlabel )

		# Set the labels:
		if labels is not None :
			if not isinstance( labels, list ) : labels = [ labels ]
			for i, ax in enumerate( self.axes ) :
				i_var = sum( n_var[:i] )
				if n_var[i] > 1 :
					i_end = min( len( labels ), i_var + n_var[i] )
					ax.legend( self._lines[i_var:i_end], labels[i_var:i_end] )
				elif len( labels ) > i_var :
					ax.set_ylabel( labels[i_var] )

		# Set the titles:
		if titles is not None :
			if not isinstance( titles, list ) : titles = [ titles ]
			for title, ax in zip( titles, self.axes ) :
				ax.set_title( title )

		# Set logarithmic scale for the specified subplot(s):
		if log :
			for i, ax in enumerate( self.axes ) :
				if log is True or i + 1 in log :
					ax.set_yscale( 'symlog' )

		# Set the interactive mode so that the figure can be displayed without blocking:
		plt.ion()
		plt.show()

		self.x_step = x_step

		if keep :
			def keep_figure_open() :
				plt.ioff()
				plt.show()
			atexit.register( keep_figure_open )
	
	def append( self, new_ydata, new_xdata=None ) :

		if not isinstance( new_ydata, collections.Iterable ) : new_ydata = [ new_ydata ]

		if len( new_ydata ) != len( self._ydata ) :
			raise ValueError( 'Wrong argument new_ydata: len( new_ydata ) = %i the number of variables to plot is %i'
			                   % ( len( new_ydata ), len( self._ydata ) ) )

		if new_xdata is not None :
			x_next = new_xdata
		elif self._xdata :
			x_next = self._xdata[-1] + self.x_step
		else :
			x_next = self.x_step

		self._xdata.append( x_next )
		for ydata, new_value in zip( self._ydata, new_ydata ) :
			ydata.append( new_value )

		self._update_figure()

	def _update_figure( self ) :

		for i, line in enumerate( self._lines ) :
			line.set_xdata( self._xdata )
			line.set_ydata( self._ydata[i] )

		for ax in self.axes :
			ax.relim()
			ax.autoscale()

		try :
			self.fig.canvas.draw()
			self.fig.canvas.flush_events()
		except :
			pass
	
	def close( self ) :

		plt.close( self.fig )
