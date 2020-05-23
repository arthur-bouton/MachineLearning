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
import atexit
import sys


class Monitor :
	""" 
	Plot variables incrementally.

	Author: Arthur Bouton [arthur.bouton@gadz.org]
	"""

	def __init__( self, title=None, label=None, log=False, keep=True, x_step=1 ) :

		self.fig, self.axes = plt.subplots()
		if title is not None :
			self.fig.canvas.set_window_title( title )
		else :
			self.fig.canvas.set_window_title( sys.argv[0] )
		self.axes.set_title( label )
		if log :
			self.axes.set_yscale( "symlog" )
		self.axes.grid( True )
		self._xdata = []
		self._ydata = []
		self._lines = self.axes.plot( [] )
		plt.ion()
		plt.show()

		self.x_step = x_step

		if keep :
			def keep_figure_open() :
				plt.ioff()
				plt.show()
			atexit.register( keep_figure_open )
	
	def append( self, ydata, xdata=None ) :

		if xdata is not None :
			x_next = xdata
		elif self._xdata :
			x_next = self._xdata[-1] + self.x_step
		else :
			x_next = self.x_step

		self._xdata.append( x_next )
		self._ydata.append( ydata )
		
		self._lines[0].set_xdata( self._xdata )
		self._lines[0].set_ydata( self._ydata )

		self.axes.relim()
		self.axes.autoscale()

		self.fig.canvas.draw()
		self.fig.canvas.flush_events()
	
	def close( self ) :

		plt.close( self.fig )
