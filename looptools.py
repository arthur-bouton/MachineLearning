""" 
Tools to monitor, extract data and have control during algorithm progress loops.

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
import sys


class Monitor :
	""" 
	A class to plot variables incrementally on a persistent figure.

	Parameters
	----------
	n_var : int or iterable of ints, optional, default: 1
		The number of variables to be plotted on each subplot.
		If a unique integer is given, there will be only one graph.
		Otherwise, the length of the iterable defines the number of subplots.
	labels :  str or list of strs, optional, default: None
		Labels for each variable to be plotted.
		If a subplot contains several variables, it will be displayed as a
		legend. Otherwise, it will be a label on the y-axis.
		If a unique string is given, it will be applied to the first variable.
	titles : str or list of strs, optional, default: None
		Titles for each subplot.
		If a unique string is given, it will be applied to the first subplot.
	xlabel : str, optional, default: None
		Label for the x-axis.
	name : str, optional, default: None
		Title used by the figure window.
		If None (default), the name of the calling script is used.
	log : bool, int or iterable of ints, optional, default: False
		Whether to use logarithmic scales.
		If True, all subplots will do.
		If a unique integer is given, only the nth subplot will do.
		If a list is provided, each integer specifies the subplots
		requiring a logarithmic scale.
	keep : bool, optional, default: True
		Whether to make the figure persistent after the end of the script.
		True by default.
	xstep : int or float, optional, default: 1
		The default gap to use between each x-axis value when adding new data.
		It is ignored when the method `add_data` is called with its second argument.
	datamax : int, optional, default: None
		The maximum amount of consecutive data to store and display.
		Past this limit, oldest data are scrapped.
		If None (default), all the data are kept until the method `clear` is called.
	zero : bool, int or iterable of ints, optional, default: False
		Whether to keep the zero axis in sight when adjusting the bounding boxes.
		If True, all subplots will do.
		If a unique integer is given, only the nth subplot will do.
		If a list is provided, each integer specifies the subplots
		that will keep the zero axis in their bounding box.
	plot_kwargs : dict of dicts, optional, default: None
		A dictionary containing dictionaries of keyword arguments to be passed
		to the calls to `matplotlib.axes.Axes.plot`.
		Each key has to be an integers corresponding to the number of the variable
		the dictionary of arguments is to applied to.
		If the key is 0, the dictionary is applied to every plot.
	
	Example
	-------
	To plot two variables alpha and beta on a same graph and a third variable gamma on a second graph below using a dashed line, do:

		monitor = Monitor( [ 2, 1 ], titles=[ 'First graph', 'Second graph' ], labels=[ '$\\alpha$', '$\\beta$', '$\gamma$' ], plot_kwargs={3: {'ls':'--'}} )

		for i in range( 100 ) :

			alpha = i**2
			beta = i**3
			gamma = 1/( 1 + i )

			monitor.add_data( alpha, beta, gamma )

	"""

	def __init__( self, n_var=1, labels=None, titles=None, xlabel=None, name=None, log=False, keep=True, xstep=1, datamax=None, zero=False, plot_kwargs=None ) :

		if not isinstance( n_var, collections.Iterable ) : n_var = [ n_var ]

		self._nvar = n_var
		self.xstep = xstep
		self.datamax = datamax
		self.zero = zero

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
		for i, ax in enumerate( self.axes ) :
			if n_var[i] < 1 :
				raise ValueError( 'Wrong argument n_var: there cannot be less than one variable on a subplot' )
			for j in range( n_var[i] ) :

				self._ydata.append( [] )

				# Add extra plotting options:
				kwargs = {}
				if plot_kwargs is not None :
					if 0 in plot_kwargs :
						kwargs.update( plot_kwargs[0] )
					i_line = sum( n_var[:i] ) + j + 1
					if i_line in plot_kwargs :
						kwargs.update( plot_kwargs[i_line] )

				self._lines.append( ax.plot( [], **kwargs )[0] )

			ax.grid( ls='dotted', alpha=0.8 )
			ax.abscissa = ax.axhline( ls='dashed', alpha=0.2 )
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
			import atexit
			import os

			def keep_figure_open() :

				def handler( signum, frame ) :
					sys.stderr.write( '\r' )
					os._exit( os.EX_OK )

				signal.signal( signal.SIGINT, handler )

				plt.ioff()
				plt.show()

			atexit.register( keep_figure_open )
	
	def add_data( self, *new_data ) :
		""" 
		Add new data to the figure.

		Parameters
		----------
		new_data : float or iterable of floats
			Each argument is the value or list of next successive values to add to the corresponding variable
			in their plot order.
			Optionally, the corresponding x-axis value(s) can be specified as first argument. If not specified,
			`xstep` is used as a gap between each x-axis value (1 by default).
		"""

		# Check the number of arguments provided:
		n_y = len( self._ydata )
		if len( new_data ) != n_y and len( new_data ) != n_y + 1 :
			raise ValueError( 'Wrong amount of arguments for Monitor.add_data: expecting %i or %i arguments but received %i'
			                   % ( n_y, n_y + 1, len( new_data ) ) )

		# Make new_data a list of iterables:
		new_data = [ [ values ] if not isinstance( values, collections.Iterable ) else values for values in new_data ]

		# Check the length of every argument:
		for values in new_data :
			if len( values ) != len( new_data[0] ) :
				raise ValueError( 'Wrong arguments for Monitor.add_data: the data do not have all the same length' )

		# Characterize the x-axis and y-axis values:
		if len( new_data ) == n_y + 1 :
			new_x_values = new_data[0]
			new_y_values = new_data[1:]
		else :
			new_x_values = [ self._xdata[-1] + self.xstep if self._xdata else self.xstep ]
			for _ in range( len( new_data[0] ) - 1 ) :
				new_x_values.append( new_x_values[-1] + self.xstep )
			new_y_values = new_data

		# Add the new data:
		self._xdata.extend( new_x_values )
		for ydata, values in zip( self._ydata, new_y_values ) :
			ydata.extend( values )

		self._update_figure()

	def _update_figure( self ) :

		# Trim the data if required:
		if self.datamax is not None and len( self._xdata ) > self.datamax :
			trim_start = len( self._xdata ) - self.datamax
			del self._xdata[0:trim_start]
			for ydata in self._ydata :
				del ydata[0:trim_start]

		# Update the plotted data:
		for i, line in enumerate( self._lines ) :
			line.set_xdata( self._xdata )
			line.set_ydata( self._ydata[i] )

		# Adjust the bounds of the boxes:
		i_line = 0
		for i_ax, ax in enumerate( self.axes ) :
			if self.zero is True or self.zero == i_ax + 1 or isinstance( self.zero, collections.Iterable ) and i_ax + 1 in self.zero :
				# Keep the zero axis in sight:
				ax.relim()
				i_line += self._nvar[i_ax]
			else :
				# Exclude the zero axis:
				first = True
				for _ in range( self._nvar[i_ax] ) :
					ax.dataLim.update_from_path( self._lines[i_line].get_path(), first )
					first = False
					i_line += 1
			ax.autoscale()

		# Update the figure without throwing an error if the window has been closed:
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
		data : a tuple of lists of floats
			The first list contains the x-axis values and is followed by the lists of values for each variable plotted.
		"""

		return ( self._xdata, *self._ydata )
	
	def clear( self ) :
		""" Clear the data and the figure. """

		self._xdata = []
		self._ydata = [ [] for _ in self._ydata ]

		self._update_figure()

	def close( self ) :
		""" Close the figure. """

		plt.close( self.fig )

	def __len__( self ) :

		return len( self._xdata )

	def __getitem__( self, key ) :

		return ( self._xdata[key], *( y[key] for y in self._ydata ) )

	def __setitem__( self, key, data ) :

		if isinstance( data, tuple ) and len( data ) == len( self._ydata ) + 1 :
			self._xdata[key] = data[0]
			y_values = data[1:]
		elif not isinstance( data, collections.Iterable ) and isinstance( key, slice ) :
			y_values = [ [ data ]*len( self._xdata[key] ) ]
		elif not isinstance( data, tuple ) :
			y_values = [ data ]
		else :
			y_values = data

		if len( y_values ) == 1 and len( self._ydata ) > 1 :
			y_values = [ y_values[0] ]*len( self._ydata )
		elif len( y_values ) != len( self._ydata ) :
			raise ValueError( 'Wrong amount of right side elements: expecting 1, %i or %i elements but received %i'
			                   % ( len( self._ydata ), len( self._ydata ) + 1, len( y_values ) ) )

		for i, y in enumerate( self._ydata ) :
			y[key] = y_values[i]

		self._update_figure()

	def __delitem__( self, key ) :

		del self._xdata[key]
		for i, y in enumerate( self._ydata ) :
			del y[key]

		self._update_figure()

	def __call__( self, *data ) :
		"""
		When called directly, the data are replaced with the new ones.
		It can be used to slice the data plotted by doing for example:

			monitor( *monitor[a:b] )

		Which is equivalent to:

			del monitor[:a]
			del monitor[b-a:]
		"""

		self._xdata = []
		self._ydata = [ [] for _ in self._ydata ]

		self.add_data( *data )



def strange( input_string, range_char='-' ) :
	"""
	Create a list of integers from a string describing a series of ranges.

	Parameters
	----------
	input_string : str
		The string describing the list of integers to output.
		Integers or ranges to concatenate are separated by commas.
		A dash, or the character specified by the argument 'range_char',
		defines the two bounds of a range.
		An optional slash followed by an integer after a range specifies
		the step of the range.
	range_char : str, default '-'
		The character that is used to split the two bounds of each range.
		It is required to change it in order to work with negative values.

	Examples
	--------
	strange( '5,2-6/2,1-3' )  -> [5, 2, 4, 6, 1, 2, 3]
	strange( '6-2/2' )        -> [6, 4, 2]
	strange( '-2_-6/2', '_' ) -> [-2, -4, -6]
	"""

	output_list = []

	for range_string in input_string.split( ',' ) :

		range_step_list = range_string.split( '/' )
		if len( range_step_list ) == 2 :
			step = abs( int( range_step_list[1] ) )
		elif len( range_step_list ) > 2 :
			raise ValueError( "There are too many '/' for the range described by %s" % range_string )
		else :
			step = 1

		range_list = range_step_list[0].split( range_char )
		if len( range_list ) == 2 :
			start = int( range_list[0] )
			stop  = int( range_list[1] )
			order = 1 if start <= stop else -1
			range_list = range( start, stop + order, order*step )
			output_list.extend( range_list )
		else :
			output_list.append( int( range_step_list[0] ) )

	return output_list



import re


class Datafile :
	"""
	A class to extract numerical data from a file.

	Parameters
	----------
	filename : str
		The path to the file containing the data.
	columns : list of ints or iterators, optional, default: None
		Each integer specifies the number of a column where to look for numerical data.
		The lines that don't include a numerical value in every column enumerated by 'columns'
		will be dismissed.
		The data will be output in the order specified by 'columns'.
		It can include iterators such as range.
		If None, it will assume the columns where to look for data according to the first line
		where at least one numerical value is found. In this case, it can be used in combination
		with the argument 'offset' to indicate the first line containing data, or in combination
		with the arguments 'filter' or 'ncols' to filter the irrelevant lines.
	sep : str, optional, default: ' '
		The string to be used to split the lines in columns.
	ncols : int, optional, default: None
		The exact number of columns that a line must comprise in order to be considered.
	filter : str, optional, default: None
		A regex that a line must contain in order to be considered.
	offset : int, optional, default: 0
		Offset of lines before starting to look for data.
	length : int, optional, default: None
		Maximum number of data to read.

	"""

	def __init__( self, filename, columns=None, sep=' ', ncols=None, filter=None, offset=0, length=None ) :

		self._filename = filename
		self._sep = sep
		self._ncols = ncols
		self._regex = filter
		self._offset = offset
		self._length = length
		self._nline = 0
		self._ndata = None

		self._columns = []
		if isinstance( columns, str ) :
			self._columns = [ col - 1 for col in strange( columns ) ]
		elif columns is not None :
			# Unfold the iterables as a list of integers:
			if not isinstance( columns, collections.Iterable ) :
				columns = [ columns ]
			for item in columns :
				if not isinstance( item, collections.Iterable ) :
					item = [ item ]
				for col in item :
					if not isinstance( col, int ) :
						raise ValueError( "Wrong element in the argument 'columns': expecting integers but received %s" % type( col ) )
					self._columns.append( col - 1 )
		else :
			# Identify the data fields to look for in the first line where at least one float can be found:
			with open( self._filename, 'r' ) as file :

				line = self._next_line( file, from_beginning=True )

				while line :

					for i, word in enumerate( line ) :
						try :
							float( word )
						except ValueError :
							continue
						self._columns.append( i )

					if len( self._columns ) > 0 :
						break

					line = self._next_line( file )

				if len( self._columns ) == 0 :
					raise RuntimeError( 'No numerical data field could have been identified in the file %s' % self._filename )

	def _next_line( self, file, from_beginning=False ) :

		if from_beginning :
			self._nline = 0

		while True :

			strline = file.readline()

			self._nline += 1

			if not strline :
				return False

			if self._nline <= self._offset :
				continue

			if self._regex is not None and not re.search( self._regex, strline ) :
				continue

			# Remove leading and trailing whitespaces and newlines:
			strline = strline.strip( ' \n' )

			# Remove duplicate whitespaces:
			strline = ' '.join( strline.split() )

			# Split the line into columns:
			line = strline.split( self._sep )

			if self._ncols is not None and len( line ) != self._ncols :
				continue

			return line

	def __iter__( self ) :

		self._ndata = 0

		with open( self._filename, 'r' ) as file :

			line = self._next_line( file, from_beginning=True )

			while line :

				values = []
				try :
					for col in self._columns :
						values.append( float( line[col] ) )
				except ( ValueError, IndexError ) :
					pass
				else :
					self._ndata += 1

					yield values if len( values ) > 1 else values[0]

					if self._length is not None and self._ndata >= self._length :
						break

				line = self._next_line( file )

	def __len__( self ) :

		if self._ndata is not None :
			return self._ndata
		else :
			count = 0
			for _ in self :
				count += 1

			return count

	def get_data( self ) :
		"""
		Returns all the data from a throughout scan of the file.
		Each list corresponds to a column.

		Returns
		-------
		data : a list of lists of values
			The lists of values for each series extracted from the file.

		The data returned can be plotted straight away with the class Monitor by doing for example:

			Monitor( len( data ) ).add_data( *data )

		Or in order to process the first series as the x-axis values:

			Monitor( len( data ) - 1 ).add_data( *data )
		"""

		if len( self._columns ) == 1 :
			return [ values for values in self ]

		data = [ [] for _ in self._columns ]
		for values in self :
			for col, value in zip( data, values ) :
				col.append( value )

		return data

	def get_data_by_rows( self ) :
		"""
		Returns all the data from a throughout scan of the file.
		Each list corresponds to a line read.

		Returns
		-------
		data : a list of lists of values
			The list the values for each line extracted from the file.
		"""

		return [ values for values in self ]
