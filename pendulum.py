from numpy import *
from scipy.integrate import odeint
from matplotlib.pyplot import *


class Pendulum() :
	""" 
	A simple pendulum to be controlled by a torque at its hinge.

	Author: Arthur Bouton [arthur.bouton@gadz.org]
	"""

	s_dim = 2
	a_dim = 1

	def __init__( self, initial_angle=180, store_data=False, include_stddev=False ) :
		#self.m = 0.5 #kg
		self.m = 1 #kg
		self.l = 0.3 #m
		self.k = 0.05 #N.m.s
		self.g = 9.81 #N/kg
		self.umax = 1 #N.m
		self.timestep = 0.1 #s

		self.reset( initial_angle, store_data, include_stddev )
	
	def model( self, x, t, u ) :
		#u = u/5
		u = clip( u, -self.umax, self.umax )
		ddtheta = -self.g/self.l*sin( x[0] - pi ) - self.k/self.m/self.l**2*x[1] + u/self.m/self.l**2
		#if t > 6 and t < 6.05 :
			#ddtheta += 5*umax/self.m/self.l**2
		return [ x[1], ddtheta ]

	def d_model( self, x, t, u ) :
		return [ [ 0., 1. ], [ -self.g/self.l*cos( x[0] - pi ), -self.k/self.m/self.l**2 ] ]
	
	def reset( self, initial_angle=180, store_data=False, include_stddev=False ) :
		self.x0 = array([ initial_angle*pi/180, 0*pi/180 ])
		self.t = 0.
		self.x = array( self.x0 )
		self.diff = 0.
		self.Rt = 0.

		if store_data :
			self.t_data = [ self.t ]
			self.u_data = []
			self.x_data = [ self.x*180/pi ]
			self.theta_abs = [ self.x[0] ]
			if include_stddev :
				self.u_stddev_data = []

		self.store_data = store_data
		self.include_stddev = include_stddev

		return self.get_obs()

	def step( self, u, stddev=0 ) :

		self.x = odeint( self.model, self.x, [ self.t, self.t+self.timestep ], (u,), self.d_model )[-1]
		#self.x = odeint( self.model, self.x, [ self.t, self.t+self.timestep ], (u,) )[-1]
		#self.x = self.x + self.timestep*array( self.model( self.x, self.t, u ) )
		x_int = self.x[0]
		self.x[0] = ( self.x[0] + pi )%( 2*pi ) - pi
		self.diff += x_int - self.x[0]
		self.t += self.timestep

		# Reward :
		r = -cos( self.x[0] - pi )
		#r = 25 - ( self.x[0]*pi/180 )**2 if abs( self.x[0] ) <= 5*pi/180 else 0
		self.Rt += r

		if self.store_data :
			self.t_data.append( self.t )
			self.u_data.append( u )
			self.x_data.append( self.x*180/pi )
			self.theta_abs.append( self.diff + self.x[0] )
			if self.include_stddev :
				self.u_stddev_data.append( stddev )

		return self.get_obs(), r, False, None
		#done = True if abs( self.diff + self.x[0] - pi ) > 3/4*pi else False
		#if done : r = -10
		#return self.x, r, done, None
	
	def get_obs( self ) :
		return self.x
	
	def get_Rt( self ) :
		return self.Rt/( self.t/self.timestep + 1 )

	def print_eval( self ) :

		Rt = self.Rt/( self.t/self.timestep + 1 )

		if self.store_data :
			success_rate = sum( [ ( 1. if abs( a[0] ) < 10 else 0. ) for a in self.x_data ] )/len( self.x_data )*100
			print( 'tf %4.1f | Rt %+7.4f | Success rate %5.1f %% | Nr %+3d' % ( self.t, Rt, success_rate, ( self.diff - self.x0[0] )/( 2* pi ) ) )
		else :
			print( 'tf %4.1f | Rt %+7.4f | Nr %+3d' % ( self.t, Rt, ( self.diff - self.x0[0] )/( 2* pi ) ) )
	
	def plot_trial( self, title=None, plot_stddev=True ) :

		if not self.store_data :
			return

		fig, ax = subplots( 3, sharex=True )
		fig.canvas.set_window_title( title if title is not None else 'Pendulum trial' )
		ax[0].set_ylabel( 'u' )
		ax[0].plot( self.t_data[:-1], self.u_data )
		if self.include_stddev and plot_stddev :
			ax[0].plot( self.t_data[:-1], array( self.u_data ) + array( self.u_stddev_data )/2, '--' )
			ax[0].plot( self.t_data[:-1], array( self.u_data ) - array( self.u_stddev_data )/2, '--' )
		ax[0].set_ylim( [ -1, 1 ] )
		ax[0].grid( True )
		ax[1].set_ylabel( u'$\\theta$' )
		ax[1].plot( self.t_data, [ x*180/pi for x in self.theta_abs ] )
		#ax[1].plot( t_data, [ x[0] for x in self.x_data ] )
		#ax[1].set_ylim( [ -180, 180 ] )
		ax[1].grid( True )
		ax[2].set_ylabel( u'$\omega$' )
		ax[2].plot( self.t_data, [ x[1] for x in self.x_data ] )
		ax[2].grid( True )
		xlabel( u'$t$' )
		xlim([ self.t_data[0], self.t_data[-1] ])

	def plot3D( self, actor, critic=None, include_stddev=False ) :

		from mpl_toolkits.mplot3d import Axes3D

		resolution = 100
		x_scale = linspace( -180, 180, resolution )
		y_scale = linspace( -4*180, 4*180, resolution )
		X, Y = meshgrid( x_scale, y_scale )
		Za = zeros( ( resolution, resolution ) )
		if critic is not None :
			Zc = zeros( ( resolution, resolution ) )
		if include_stddev :
			Zs = zeros( ( resolution, resolution ) )
		for i, x in enumerate( x_scale ) :
			for j, y in enumerate( y_scale ) :
				if critic is not None :
					Zc[j][i] = critic( array([ x*pi/180, y*pi/180 ]) )
				if include_stddev :
					u, u_stddev = actor( array([ x*pi/180, y*pi/180 ]), return_stddev=True )
					u = clip( u, -self.umax, self.umax )
					Za[j][i] = u
					Zs[j][i] = u_stddev**2
				else :
					Za[j][i] = actor( array([ x*pi/180, y*pi/180 ]) )
					#Za[j][i] = clip( actor( array([ x*pi/180, y*pi/180 ]) ), -self.umax, self.umax )

		if self.store_data and critic is not None :
			zc = []
			for x in self.x_data :
				zc.append( critic( x*pi/180 ) + 5 )
				#zc.append( critic( x*pi/180 ) )

		elev = 30 ; azim = -120

		if critic is not None :
			fig = figure( 'Critic' )
			ax = axes( projection='3d' )
			ax.plot_surface( X, Y, Zc )
			#ax.set_zlim3d( -1, 1 )
			ax.set_xlabel( '$\\theta$' )
			ax.set_ylabel( '$\omega$' )
			ax.set_zlabel( 'V' )
			ax.view_init( elev, azim )

			if self.store_data :
				ax.scatter( [ x[0] for x in self.x_data ], [ x[1] for x in self.x_data ], zc, c='r', marker='o', alpha=1 )

		if include_stddev :
			fig = figure( 'Actor variance' )
			ax = axes( projection='3d' )
			ax.plot_surface( X, Y, Zs )
			ax.set_xlabel( '$\\theta$' )
			ax.set_ylabel( '$\omega$' )
			ax.set_zlabel( 'Var(u)' )
			ax.view_init( elev, azim )

		fig = figure( 'Actor' )
		ax = axes( projection='3d' )
		ax.plot_surface( X, Y, Za )
		#ax.set_zlim3d( -1, 1 )
		ax.set_xlabel( '$\\theta$' )
		ax.set_ylabel( '$\omega$' )
		ax.set_zlabel( 'u' )
		ax.view_init( elev, azim )

	def show( self ) :
		show()


