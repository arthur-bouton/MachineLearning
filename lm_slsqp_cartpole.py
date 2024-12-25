#!/usr/bin/env python3
""" 
Automated control synthesis to swing up a cart-pole for which the physical parameters are unknown.

The parameter identification is performed by a non-linear regression using the Levenberg-Marquardt algorithm.
The trajectory planning is then based on direct trapezoidal collocation using Sequential Least Squares Programming (SLSQP).
Finally, the trajectory tracking is ensured by a finite-horizon Linear-Quadratic Regulator (LQR).

Author: Arthur Bouton [arthur.bouton@gadz.org]

"""
import numpy as np
import sys
import os
from scipy.integrate import odeint
import scipy.optimize
import LQR
from cartpole import Cartpole


# Identifier name for the data:
data_id = 'test_1'

script_name = os.path.splitext( os.path.basename( __file__ ) )[0]
data_dir = 'training_data/' + script_name + '/' + data_id



# Suggested dynamics of the system, for which the parameter values are to be determined:
n_params = 5
def f_dynamics( x, t, f, params ) :
	f = np.array( f ).item()

	g  = params[0]
	mp = params[1]
	mc = params[2]
	lp = params[3]
	kp = params[4]

	tau = -kp*x[3]

	costheta = np.cos( x[2] - np.pi )
	sintheta = np.sin( x[2] - np.pi )

	ddx = ( mp*sintheta*( lp*x[3]**2 + g*costheta ) + f - tau*costheta/lp )/( mc + mp*sintheta**2 )
	ddtheta = ( lp*mp*x[3]**2*costheta*sintheta + ( mc + mp )*( g*sintheta - tau/( lp*mp ) ) + f*costheta )/( -lp*( mc + mp*sintheta**2 ) )

	return np.array([ x[1], ddx, x[3], ddtheta ])



if len( sys.argv ) == 1 or sys.argv[1] != 'eval' :


	############################
	# PARAMETER IDENTIFICATION #
	############################

	timestep = 0.05 #s

	# Ornstein-Uhlenbeck process for random exploration:
	theta = 10
	sigma = 2
	dt = timestep
	u = 0
	def draw() :
		global u
		u = u - theta*u*timestep + sigma*np.random.randn()*np.sqrt( timestep )
		return u

	# Exploration trial:
	xdata = []
	ydata = []
	env = Cartpole( ( 0, 180 ), store_data=True, control='force' )
	for t in range( 100 ) :
		s = env.get_obs()
		f = draw()
		s_next, _, done, _ = env.step( f )
		if done : break
		xdata.append( np.array([[ *s, f ]]).T )
		ydata.append( s_next[1] + s_next[3] )
	xdata = np.hstack( xdata )
	ydata = np.array( ydata )
	env.animate( title='Exploration run used for parameter-identification (close to continue)' )


	# Function predicting the next expected state when integrating the dynamics model:
	def f_predictions( xdata, *params ) :
		y = []
		for j in range( xdata.shape[1] ) :
			x = xdata[:4,j]
			f = xdata[4,j]
			xkplus1 = odeint( f_dynamics, x, [ 0, timestep ], ( f, params ) )[-1]
			y.append( xkplus1[1] + xkplus1[3] )
		return np.array( y )

	# Non-linear regression using the Levenberg-Marquardt algorithm:
	p0 = ( 1, )*n_params
	params, pcov = scipy.optimize.curve_fit( f_predictions, xdata, ydata, p0, method='lm' )

	print( 'Result of the parameter identification:\ng:  %f\nmp: %f\nmc: %f\nlp: %f\nkp: %f' % ( *params, ) )


	#######################
	# TRAJECTORY PLANNING #
	#######################

	# Trajectory duration:
	T = 3 #s

	# Number of segments:
	N = 60

	# Path bounds:
	lc = 1. #m
	fmax = 10. #N
	vmax = 0.8 #m/s


	dt_traj = T/N #s

	lp = params[3]
	def trajectory_cost( x ) :
		xk = [ x[5*i:5*i+4] for i in range( N + 1 ) ]
		cost = sum( np.sqrt( ( x[0] - lp*np.sin( x[2] ) )**2 + ( lp*np.cos( x[2] ) - lp )**2 ) for x in xk )
		fk = [ x[5*i+4] for i in range( N + 1 ) ]
		cost += 0.1*sum( fk[i]**2*dt_traj for i in range( N + 1 ) ) # ZOH
		#cost += 0.1*sum( ( fk[i]**2 + fk[i+1]**2 )/2*dt_traj for i in range( N ) ) # FOH
		return cost

	def dynamics_constraints( x ) :
		xk = [ x[5*i:5*i+4] for i in range( N + 1 ) ]
		fk = [ x[5*i+4] for i in range( N + 1 ) ]
		eqs = []
		for i in range( N ) :
			eqs.extend( xk[i+1] - xk[i] - ( f_dynamics( xk[i+1], 0, fk[i], params ) + f_dynamics( xk[i], 0, fk[i], params ) )/2*dt_traj ) # ZOH
			#eqs.extend( xk[i+1] - xk[i] - ( f_dynamics( xk[i+1], 0, fk[i+1], params ) + f_dynamics( xk[i], 0, fk[i], params ) )/2*dt_traj ) # FOH
		return eqs

	initial_state = lambda x: x[:4] - np.array([ 0, 0, np.pi, 0 ])
	final_state = lambda x: x[-5:-1] - np.array([ 0, 0, 0, 0 ])
	#final_state = lambda x: x[-4:-1] - np.array([ 0, 0, 0 ])

	constraints = []
	constraints.append( { 'type': 'eq', 'fun': dynamics_constraints } )
	constraints.append( { 'type': 'eq', 'fun': initial_state } )
	constraints.append( { 'type': 'eq', 'fun': final_state } )

	# Path bounds:
	x_bounds = ( ( -lc, lc ), ( -vmax, vmax ), ( None, None ), ( None, None ) )
	f_bounds = ( -fmax, fmax )
	bounds = ( x_bounds + (f_bounds,) )*( N + 1 )

	# Initial guess:
	x0 = np.zeros( 5*( N + 1 ) )
	for i in range( N + 1 ) :
		x0[5*i+2] = np.pi*( 1 - i/N )
	
	iteration = 0
	def callback( xk ) :
		global iteration
		iteration += 1
		print( '\rIteration %i: cost = %f ' % ( iteration, trajectory_cost( xk ) ), end='' )

	print( 'Trajectory optimization:' )
	import time, datetime
	start = time.time()

	# Trajectory optimization using non-linear programming:
	options = { 'maxiter': 1000 }
	res = scipy.optimize.minimize( trajectory_cost, x0, method='SLSQP', bounds=bounds, constraints=constraints, options=options, callback=callback )
	end = time.time()
	print( '\n' + res.message )
	print( 'nit: %i nfev: %i fun: %f time: %s' % ( res.nit, res.nfev, res.fun, str( datetime.timedelta( seconds=end-start ) ) ) )
	if not res.success :
		exit( -1 )


	x_list = [ res.x[5*i:5*i+4] for i in range( N + 1 ) ]
	f_list = [ res.x[5*i+4] for i in range( N + 1 ) ]

	#from looptools import Monitor
	#trajplot = Monitor( [ 1, 1, 1, 1 ], labels=[ '$x$', '$\dot{x}$', '$\\theta$', '$\dot{\\theta}$' ], keep=False )
	#for t, xk in zip( np.linspace( 0, T, N + 1 ), x_list ) :
		#trajplot.add_data( t, *xk, update=False )
	#trajplot.update()
	#Monitor( 1, labels='$f$', keep=False ).add_data( np.linspace( 0, T, N + 1 ), f_list )

	answer = input( 'Save the trajectory in ' + data_dir + '? (y) ' )
	if answer.strip() == 'y' :
		os.makedirs( data_dir, exist_ok=True )
		np.save( data_dir + '/x_sequence', x_list )
		np.save( data_dir + '/f_sequence', f_list )
		np.save( data_dir + '/dt', dt_traj )
		np.save( data_dir + '/dynamics_parameters', params )
		print( 'Trajectory saved.' )


if len( sys.argv ) > 1 and sys.argv[1] == 'eval' :
	x_list = np.load( data_dir + '/x_sequence.npy' )
	f_list = np.load( data_dir + '/f_sequence.npy' )
	dt_traj = np.load( data_dir + '/dt.npy' )
	params = np.load( data_dir + '/dynamics_parameters.npy' )


#######################
# TRAJECTORY TRACKING #
#######################

# Feedback-control period:
dt_control = 0.05 #s
#dt_control = 0.01 #s

# Tracking cost matrices:
Q = np.diag([ 1, 1, 1, 1 ])
R = 0.1

def linearized_dynamics( x, f, params ) :

	A = []
	for i in range( len( x ) ) :
		A.append( scipy.optimize.approx_fprime( x, lambda x: f_dynamics( x, 0, f, params )[i], 1e-6 ) )
	A = np.vstack( A )

	B = []
	for i in range( len( x ) ) :
		B.append( scipy.optimize.approx_fprime( np.array([ f ]), lambda f: f_dynamics( x, 0, f, params )[i], 1e-6 ) )
	B = np.vstack( B )

	return LQR.discretize_system_ZOH( A, B, dt_control ) # ZOH
	#return LQR.discretize_system_FOH( A, B, dt_control ) # FOH

K = LQR.dlqr_traj( [ linearized_dynamics( x, f, params ) for x, f in zip( x_list, f_list ) ], Q, R )


#########
# TRIAL #
#########

env = Cartpole( ( x_list[0][0], x_list[0][2]*180/np.pi ), store_data=True, angle='absolute', control='force' )
env.timestep = dt_control
isteps = int( dt_traj/dt_control )

err_data = []
f_data = []
x = env.get_obs()
for i in range( len( f_list ) - 1 ) :
	for j in range( isteps ) :
		ud = f_list[i] # ZOH
		#ud = f_list[i] + ( f_list[i+1] - f_list[i] )*j/isteps # FOH
		xd = x_list[i] + ( x_list[i+1] - x_list[i] )*j/isteps
		u = ud - K[i].dot( x - xd )
		#u = ud # Open loop
		err_data.append( x - xd )
		f_data.append( u )
		_, _, ep_done, _ = env.step( u.item() )
		x = env.get_obs()
		if ep_done : break
	if ep_done : break


import matplotlib.pyplot as plt

x_list = [ x_list[i] + ( x_list[i+1] - x_list[i] )*j/isteps for i in range( len( x_list ) - 1 ) for j in range( isteps ) ] + [ x_list[-1] ]
x_list = x_list[:len( env.x_data )]

fig, ax = plt.subplots( 4, sharex=True )
fig.canvas.manager.set_window_title( 'Trajectory' )
ax[0].plot( env.t_data, [ xd[0] for xd in x_list ] )
ax[0].plot( env.t_data, [ x[0] for x in env.x_data ] )
ax[0].legend( [ u'$x_{traj}$', u'$x_{trial}$' ] )
ax[1].plot( env.t_data, [ xd[1] for xd in x_list ] )
ax[1].plot( env.t_data, [ x[1] for x in env.x_data ] )
ax[1].legend( [ u'$\dot{x}_{traj}$', u'$\dot{x}_{trial}$' ] )
ax[2].plot( env.t_data, [ xd[2]*180/np.pi for xd in x_list ] )
ax[2].plot( env.t_data, [ x[2]*180/np.pi for x in env.x_data ] )
ax[2].legend( [ u'$\\theta_{traj}$', u'$\\theta_{trial}$' ] )
ax[3].plot( env.t_data, [ xd[3] for xd in x_list ] )
ax[3].plot( env.t_data, [ x[3] for x in env.x_data ] )
ax[3].legend( [ u'$\dot{\\theta}_{traj}$', u'$\dot{\\theta}_{trial}$' ] )
for i in range( 4 ) :
	ax[i].grid( True )

fig, ax = plt.subplots()
fig.canvas.manager.set_window_title( 'Control' )
ax.plot( env.t_data[:-1], [ fd for fd in f_list for j in range( isteps ) ][:len( env.x_data )-1] ) # ZOH
#ax.plot( env.t_data[:-1], [ f_list[i] + ( f_list[i+1] - f_list[i] )*j/isteps for i in range( len( f_list ) - 1 ) for j in range( isteps ) ][:len( env.x_data )-1] ) # FOH
ax.plot( env.t_data[:-1], [ f for f in f_data ] )
ax.legend( [ u'$f_{traj}$', u'$f_{trial}$' ] )
ax.grid( True )

fig, ax = plt.subplots( 4, sharex=True )
fig.canvas.manager.set_window_title( 'Controller gains' )
ax[0].plot( env.t_data[:-1], [ k[0][0] for k in K for j in range( isteps ) ][:len( env.x_data )-1] )
ax[1].plot( env.t_data[:-1], [ k[0][1] for k in K for j in range( isteps ) ][:len( env.x_data )-1] )
ax[2].plot( env.t_data[:-1], [ k[0][2] for k in K for j in range( isteps ) ][:len( env.x_data )-1] )
ax[3].plot( env.t_data[:-1], [ k[0][3] for k in K for j in range( isteps ) ][:len( env.x_data )-1] )
ax[0].set_ylabel( u'$K_x$' )
ax[1].set_ylabel( u'$K_\dot{x}$' )
ax[2].set_ylabel( u'$K_\\theta$' )
ax[3].set_ylabel( u'$K_\dot{\\theta}$' )
for i in range( 4 ) :
	ax[i].grid( True )

fig, ax = plt.subplots( 4, sharex=True )
fig.canvas.manager.set_window_title( 'Tracking errors' )
ax[0].plot( env.t_data[:-1], [ e[0] for e in err_data ] )
ax[1].plot( env.t_data[:-1], [ e[1] for e in err_data ] )
ax[2].plot( env.t_data[:-1], [ e[2] for e in err_data ] )
ax[3].plot( env.t_data[:-1], [ e[3] for e in err_data ] )
ax[0].set_ylabel( u'$x$' )
ax[1].set_ylabel( u'$\dot{x}$' )
ax[2].set_ylabel( u'$\\theta$' )
ax[3].set_ylabel( u'$\omega$' )
for i in range( 4 ) :
	ax[i].grid( True )


env.animate()
