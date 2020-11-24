"""
Author: Arthur Bouton [arthur.bouton@gadz.org]

"""
from numpy import *
from scipy.integrate import odeint
from matplotlib.pyplot import *


class Cartpole() :
	""" 
	A free pendulum mounted on a cart which lateral velocity is controlled.

	"""

	s_dim = 4
	a_dim = 1

	def __init__( self, initial_pos=( 0, 180 ), store_data=False, include_stddev=False, control='speed', angle='relative' ) :
		self.mp = 0.1 #kg
		self.mc = 0.1 #kg
		self.lp = 0.3 #m
		self.lc = 1. #m
		self.kp = 0.001 #N.m.s
		self.kc = 0.1 #N.s
		self.g = 9.81 #N/kg
		self.fmax = 10. #N
		self.vmax = 1. #m/s
		self.timestep = 0.05 #s

		self.force_control = False
		if control == 'force' :
			self.force_control = True

		self.absolute_angle = False
		if angle == 'absolute' :
			self.absolute_angle = True

		self.reset( initial_pos, store_data, include_stddev )
	
	def model( self, x, t, u ) :

		if self.force_control :
			f = clip( u, -self.fmax, self.fmax )
		else :
			f = self.cart_controller( u, x[1] ) - self.kc*x[1]
		tau = -self.kp*x[3]

		costheta = cos( x[2] - pi )
		sintheta = sin( x[2] - pi )

		ddx = ( self.mp*sintheta*( self.lp*x[3]**2 + self.g*costheta ) + f - tau*costheta/self.lp )/( self.mc + self.mp*sintheta**2 )
		ddtheta = ( self.lp*self.mp*x[3]**2*costheta*sintheta + ( self.mc + self.mp )*( self.g*sintheta - tau/( self.lp*self.mp ) ) + f*costheta )/( -self.lp*( self.mc + self.mp*sintheta**2 ) )

		return [ x[1], ddx, x[3], ddtheta ]

	def cart_controller( self, vd, v ) :

		vd = clip( vd, -self.vmax, self.vmax )

		f = 100*( vd - v )

		return clip( f, -self.fmax, self.fmax )
	
	def reset( self, initial_pos=( 0, 180 ), store_data=False, include_stddev=False ) :
		self.x0 = array([ initial_pos[0], 0, initial_pos[1]*pi/180, 0*pi/180 ])
		self.t = 0.
		self.x = array( self.x0 )
		self.Rt = 0.

		if store_data :
			self.t_data = [ self.t ]
			self.u_data = []
			self.x_data = [ self.x ]
			if include_stddev :
				self.v_stddev_data = []

		self.store_data = store_data
		self.include_stddev = include_stddev

		return self.get_obs()

	def step( self, u, stddev=0 ) :

		self.x = odeint( self.model, self.x, [ self.t, self.t+self.timestep ], (u,) )[-1]
		#self.x = self.x + self.timestep*array( self.model( self.x, self.t, u ) )
		self.t += self.timestep

		# Reward :
		if abs( self.x[0] ) >= self.lc :
			r = -10
			terminal = True
		else :
			r = -sqrt( ( self.x[0] - self.lp*sin( self.x[2] ) )**2 + ( self.lp*cos( self.x[2] ) - self.lp )**2 )
			r /= 2*self.lp
			terminal = False
		self.Rt += r

		if self.store_data :
			self.t_data.append( self.t )
			self.u_data.append( u )
			self.x_data.append( self.x )
			if self.include_stddev :
				self.v_stddev_data.append( stddev )

		return self.get_obs(), r, terminal, None
	
	def get_obs( self ) :
		if self.absolute_angle :
			return self.x
		else :
			x = array( self.x )
			x[2] = pi - ( pi - x[2] )%( 2*pi )
			return x
	
	def get_Rt( self ) :
		return self.Rt/( self.t/self.timestep + 1 )

	def print_eval( self ) :

		Rt = self.Rt/( self.t/self.timestep + 1 )

		if self.store_data :
			success_rate = sum( [ ( 1. if sqrt( ( x[0] - self.lp*sin( x[2] ) )**2 + ( self.lp*cos( x[2] ) - self.lp )**2 ) < self.lp/2 else 0. )
			                      for x in self.x_data ] )/len( self.x_data )*100
			print( 'tf %4.1f | Rt %+7.4f | Success rate %5.1f %% | Nr %+3d' % ( self.t, Rt, success_rate, int( ( self.x[2] - self.x0[2] )/( 2*pi ) ) ) )
		else :
			print( 'tf %4.1f | Rt %+7.4f | Nr %+3d' % ( self.t, Rt, int( ( self.x[2] - self.x0[2] )/( 2*pi ) ) ) )

	def plot_trial( self, title=None, plot_stddev=True ) :

		if not self.store_data :
			return

		fig, ax = subplots( 4, sharex=True )
		fig.canvas.set_window_title( title if title is not None else 'Cartpole trial (state)' )
		ax[0].set_ylabel( u'$u$' )
		ax[0].plot( self.t_data[1:], self.u_data, 'r' )
		ax[0].plot( self.t_data, [ x[1] for x in self.x_data ] )
		if self.force_control :
			ax[0].legend( [ u'$f$', u'$\dot{x}$' ] )
		else :
			ax[0].legend( [ u'$v_d$', u'$\dot{x}$' ] )
		if self.include_stddev and plot_stddev :
			ax[0].plot( self.t_data[1:], array( self.u_data ) + array( self.v_stddev_data )/2, 'c--' )
			ax[0].plot( self.t_data[1:], array( self.u_data ) - array( self.v_stddev_data )/2, 'y--' )
		if not self.force_control :
			ax[0].set_ylim( [ -self.vmax, self.vmax ] )
		ax[0].grid( True )
		ax[1].set_ylabel( u'$x$' )
		ax[1].plot( self.t_data, [ x[0] for x in self.x_data ] )
		ax[1].grid( True )
		ax[2].set_ylabel( u'$\\theta$' )
		ax[2].plot( self.t_data, [ x[2]*180/pi for x in self.x_data ] )
		#ax[2].set_ylim( [ -180, 180 ] )
		ax[2].grid( True )
		ax[3].set_ylabel( u'$\omega$' )
		ax[3].plot( self.t_data, [ x[3]*180/pi for x in self.x_data ] )
		ax[3].grid( True )
		xlabel( u'$t$' )
		xlim([ self.t_data[0], self.t_data[-1] ])

	def show( self ) :
		show()
	
	def animate( self, file_path=None, title=None ) :

		if not self.store_data :
			return

		# Ensure that the interactive mode is off:
		ioff()

		from matplotlib.animation import FuncAnimation

		lp = self.lp
		lc = self.lc

		track_color = 'b'
		target_color = 'r'
		cart_color = 'k'
		pole_color = 'k'
		hole_color = 'w'
		cart_width = 0.1
		cart_height = 0.05
		stem_width = 2
		hole_radius = 0.02
		tip_radius = 0.02
		x_margin = lp
		y_margin = 0.2

		fig = figure( title if title is not None else 'Cartpole trial (animation)', figsize=( 10, 4.5 ) )
		ax = fig.add_subplot( 1, 1, 1, aspect='equal' )
		ax.plot( [ -lc, lc ], [ 0, 0 ], color=track_color )
		ax.plot( [ 0, 0 ], [ -lp*2, lp*2 ], color=target_color )
		ax.set_xlim( [ -lc - x_margin, lc + x_margin ] )
		ax.set_ylim( [ -lp - y_margin, lp + y_margin ] )

		s = self.x_data[0]

		stem, = ax.plot( [ s[0], s[0] - ( lp - tip_radius )*sin( s[2] ) ], [ 0, ( lp - tip_radius )*cos( s[2] ) ], color=pole_color, lw=stem_width )
		cart = Rectangle( ( s[0] - cart_width/2, -cart_height/2 ), cart_width, cart_height, fc=cart_color, ec=None )
		hole = Circle( ( s[0], 0 ), radius=hole_radius, fc=hole_color )
		tip = Circle( ( s[0] - lp*sin( s[2] ), lp*cos( s[2] ) ), radius=tip_radius, fc=pole_color, ec=None )

		def init() :
			artists = []
			artists.append( stem )
			artists.append( ax.add_artist( cart ) )
			artists.append( ax.add_artist( hole ) )
			artists.append( ax.add_artist( tip ) )
			return artists
		
		def update( i ) :
			s = self.x_data[i]
			stem.set_data( [ s[0], s[0] - ( lp - tip_radius )*sin( s[2] ) ], [ 0, ( lp - tip_radius )*cos( s[2] ) ] )
			cart.set_xy(( s[0] - cart_width/2, -cart_height/2 ))
			hole.center = ( s[0], 0 )
			tip.center = s[0] - lp*sin( s[2] ), lp*cos( s[2] )
			return cart, hole, stem, tip

		anim = FuncAnimation( fig, update, blit=True, init_func=init, frames=len( self.x_data ), interval=self.timestep*1e3, repeat_delay=0 )

		if file_path is not None :
			anim.save( file_path, writer='imagemagick' )
		else :
			show()
