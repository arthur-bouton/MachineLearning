""" 
Tools to monitor and have control during algorithm progress loops.

Author: Arthur Bouton [arthur.bouton@gadz.org]
"""
import signal


class Loop_handler :
	""" 
	Context manager which allows the SIGINT signal to be processed asynchronously.

	Example
	-------
	with Loop_handler() as interruption :
		while not interruption() :
			
			(do something)

			if interruption() :
				break

			(do something)

	"""

	def __init__( self ) :
		self._end = False
	
	def check_interruption( self, reset=False ) :
		state = self._end
		if reset :
			self._end = False
		return state

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
	A class to plot variables incrementally on a persistent figure.

	Parameters
	----------
	n_var : integer or iterable of integers, optional, default: 1
		The number of variables to be plotted on each subplot.
		If a unique integer is given, there will be only one graph.
		Otherwise, the length of the iterable defines the number of subplots.
	labels :  string or list of string, optional, default: None
		Labels for each variable to be plotted.
		If a subplot contains several variables, it will be displayed as a
		legend. Otherwise, it will be a label on the y-axis.
		If a unique string is given, it will be applied to the first variable.
	titles : string or list of string, optional, default: None
		Titles for each subplot.
		If a unique string is given, it will be applied to the first subplot.
	xlabel : string, optional, default: None
		Label for the x-axis.
	name : string, optional, default: None
		Title used by the figure window.
		If None (default), the name of the calling script is used.
	log : boolean, integer or iterable of integers, optional, default: False
		Whether to use logarithmic scales.
		If True, all subplots will do.
		If a unique integer is given, only the nth subplot will do.
		If a list is provided, each integer specifies the subplots
		requiring a logarithmic scale.
	keep : boolean, optional, default: True
		Whether to make the figure persistent after the end of the script.
		True by default.
	xstep : integer or float, optional, default: 1
		The default gap to use between each x-axis value when adding new data.
		It is ignored when the method `add_data` is called with its second argument.
	datamax : integer, optional, default: None
		The maximum amount of consecutive data to store and display.
		Past this limit, oldest data are scrapped.
		If None (default), all the data are kept until the method `clear` is called.
	plot_kwargs : dictionary or iterable of dictionaries, optional, default: None
		A dictionary of keyword arguments to be passed to every call of
		`matplotlib.axes.Axes.plot` or an iterable of dictionaries for each
		variable to plot (can be an empty dictionary).
	
	Example
	-------
	To plot two variables alpha and beta on a same graph, with beta using a dashed line, and a third variable gamma on a second graph below, do:

		graph = Monitor( [ 2, 1 ], titles=[ 'First graph', 'Second graph' ], labels=[ '$\\alpha$', '$\\beta$', '$\gamma$' ], plot_kwargs=[{},{'ls': '--'}] )

		for i in range( 100 ) :

			alpha = i**2
			beta = i**3
			gamma = 1/( 1 + i )

			graph.add_data( [ alpha, beta, gamma ] )

	"""

	def __init__( self, n_var=1, labels=None, titles=None, xlabel=None, name=None, log=False, keep=True, xstep=1, datamax=None, plot_kwargs=None ) :

		self.xstep = xstep
		self.datamax = datamax

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

				if isinstance( plot_kwargs, collections.Iterable ) :
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
				if log is True or log == i + 1 or isinstance( log, collections.Iterable ) and i + 1 in log :
					ax.set_yscale( 'symlog' )

		# Set the interactive mode so that the figure can be displayed without blocking:
		plt.ion()
		plt.show()

		# Create the window:
		self._update_figure()

		# Make the window persistent:
		if keep :
			def keep_figure_open() :
				plt.ioff()
				plt.show()
			atexit.register( keep_figure_open )
	
	def add_data( self, new_ydata, new_xdata=None ) :
		""" 
		Add new data to the figure.

		Parameters
		----------
		new_ydata : value, iterable of values or iterable of iterables of values
			The list of the next values to add to each variable to plot or a list of lists of successive values
			to add to each of these variables.
			If there is only one variable to be plotted, new_ydata can be interpreted directly as its next value
			or the list of its next successive values.
		new_xdata : value or iterable of values, optional, default: None
			The x-axis value for each successive value provided by new_ydata, regardless of the number of variables.
			If None (default), `xstep` is used as a gap between each x-axis value (1 by default).
		"""

		if len( self._ydata ) == 1 and ( not isinstance( new_ydata, collections.Iterable )
		                              or not isinstance( new_ydata[0], collections.Iterable ) ) :
			new_ydata = [ new_ydata ]

		if not isinstance( new_ydata[0], collections.Iterable ) :
			new_ydata = [ [ y ] for y in new_ydata ]

		if len( new_ydata ) != len( self._ydata ) :
			raise ValueError( 'Wrong argument new_ydata: len( new_ydata ) = %i while the number of variables to plot is %i'
			                   % ( len( new_ydata ), len( self._ydata ) ) )

		for var in new_ydata :
			if not isinstance( var, collections.Iterable ) or len( var ) != len( new_ydata[0] ) :
				raise ValueError( 'Wrong argument new_ydata: the data do not have all the same length' )

		if new_xdata is not None :
			if not isinstance( new_xdata, collections.Iterable ) : new_xdata = [ new_xdata ]

			if len( new_xdata ) != len( new_ydata[0] ) :
				raise ValueError( 'Wrong argument new_xdata: len( new_xdata ) = %i while there is %i new data to be added'
				                   % ( len( new_xdata ), len( new_ydata[0] ) ) )
		else :
			new_xdata = [ self._xdata[-1] + self.xstep if self._xdata else self.xstep ]
			for _ in range( len( new_ydata[0] ) - 1 ) :
				new_xdata.append( new_xdata[-1] + self.xstep )

		# Add the new data:
		self._xdata.extend( new_xdata )
		for ydata, new_values in zip( self._ydata, new_ydata ) :
			ydata.extend( new_values )

		self._update_figure()

	def _update_figure( self ) :
		
		if self.datamax is not None and len( self._xdata ) > self.datamax :
			trim_start = len( self._xdata ) - self.datamax
			del self._xdata[0:trim_start]
			for ydata in self._ydata :
				del ydata[0:trim_start]

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
	
	def get_data( self ) :
		"""
		Return the data used to plot the current figure.
		It can be used to modify or restore the data plotted by doing for example:

			data = monitor.get_data()
			monitor.clear()
			monitor.add_data( *data )

		Returns
		-------
			( ydata, xdata )
			ydata : list of list of values for each variable.
			xdata : list values for the x-axis.
		"""

		return self._ydata, self._xdata
	
	def clear( self ) :
		""" Clear the data and the figure. """

		self._xdata = []
		self._ydata = [ [] for _ in self._ydata ]

		self._update_figure()
	
	def close( self ) :
		""" Close the figure. """

		plt.close( self.fig )
