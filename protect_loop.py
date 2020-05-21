import signal


class Protect_loop :
	""" 
	Context manager which allows the SIGINT signal to be processed asynchronously.

	Author: Arthur Bouton [arthur.bouton@gadz.org]
	"""

	def __init__( self ) :
		self._end = False

	def _handler( self, signum, frame ) :
		self._end = True
	
	def check_interruption( self ) :
		return self._end

	def __enter__( self ) :
		self._original_handler = signal.getsignal( signal.SIGINT )
		signal.signal( signal.SIGINT, self._handler )
		return self.check_interruption

	def __exit__( self, type, value, traceback ) :
		signal.signal( signal.SIGINT, self._original_handler )
