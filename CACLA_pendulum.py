#!/usr/bin/env python3
""" 
Implementation of the Continuous Actor Critic Learning Automaton (CACLA) [1] to balance a pendulum.

[1] Van Hasselt, Hado, and Marco A. Wiering. "Reinforcement learning in continuous action spaces."
    2007 IEEE International Symposium on Approximate Dynamic Programming and Reinforcement Learning. IEEE, 2007.

Train new networks by calling this script without argument.
When the algorithm has converged, send SIGINT to stop it and save the network parameters by answering "y" when prompted to.
Then run the script with the word "eval" as first argument in order to evaluate the obtained policy.
The training can be took up with saved network parameters by calling the script with the word "load" as first argument.

Author: Arthur Bouton [arthur.bouton@gadz.org]
"""
from numpy import *
from scipy.integrate import odeint
from matplotlib.pyplot import *
from mpl_toolkits.mplot3d import Axes3D
from neural_networks import RBF, MLP
from looptools import Loop_handler, Monitor
import sys
import os


#sys.stdout = os.fdopen( sys.stdout.fileno(), 'w', 0 )


# Identifier name for the data:
data_id = 'rbf_1'
#data_id = 'mlp_1'

script_name = os.path.splitext( os.path.basename( __file__ ) )[0]
data_dir = 'training_data/' + script_name + '/' + data_id


#########
# MODEL #
#########

m = 0.5 #kg
l = 0.3 #m
k = 0.05 #N.m.s
g = 9.81 #N/kg
umax = 1. #N.m
step = 0.1 #s

def model( x, t, u ) :
	ddtheta = -g/l*sin( x[0] - pi ) - k/m/l**2*x[1] + u/m/l**2
	#if t > 6 and t < 6.05 :
		#ddtheta += 5*umax/m/l**2
	return x[1], ddtheta

Dmodel = lambda x, t, u : [ [ 0., 1. ], [ -g/l*cos( x[0] - pi ), -k/m/l**2 ] ]


def quick_eval( control ) :

	x0 = array([ 180*pi/180, 0*pi/180 ])
	tf = 10

	t = 0.
	x = array( x0 )
	Rt = 0.
	t_data = [ t ]
	x_data = [ x*180/pi ]
	diff = 0.

	while t < tf - step/2 :

		# Action selection:
		u = control( x )

		# Simulation step:
		x = odeint( model, x, [ t, t+step ], (u,), Dmodel )[-1]
		x_int = x[0]
		x[0] = ( x[0] + pi )%( 2*pi ) - pi
		diff += x_int - x[0]
		t += step

		# Reward :
		Rt += -cos( x[0] - pi )

		x_data.append( x*180/pi )

	Rt /= tf/step + 1
	success_rate = sum( [ ( 1. if abs( a[0] ) < 10 else 0. ) for a in x_data ] )/len( x_data )*100

	return t, Rt, success_rate, ( diff - x0[0] )/( 2* pi )


##########################
# FUNCTION APPROXIMATORS #
##########################

scaling = lambda x : array([ x[0]/pi, clip( x[1]/( 4*pi ), -1, 1 ) ])

actor = RBF( 2, linspace( -0.8, 0.8, 9 ), sigma=0.2, lambda0=1, adaptive_stepsize=False )
#actor = MLP( [ 2, 10, 10, 1 ], lambda0=0.02, adaptive_stepsize=False, activation_function='relu' )

critic = RBF( 2, linspace( -0.8, 0.8, 9 ), sigma=0.2, lambda0=1, adaptive_stepsize=False )
#critic = MLP( [ 2, 10, 10, 1 ], lambda0=0.02, adaptive_stepsize=False, activation_function='relu' )


if len( sys.argv ) == 1 or sys.argv[1] != 'eval' :

	if len( sys.argv ) > 1 and sys.argv[1] == 'load' :
		actor.load( data_dir + '/actor' )
		critic.load( data_dir + '/critic' )


	############
	# TRAINING #
	############

	# Initial conditions:
	x0 = array([ 180*pi/180, 0*pi/180 ])

	# Training parameters:
	Ttrial = 10
	gamma = 0.7
	#prob_expl = lambda n : exp( -0.002*n )
	#prob_expl = lambda n : exp( -0.0003*n )
	prob_expl = lambda n : 0.2

	ntrial = 0
	t = 0.
	x = array( x0 )

	Rt = 0.
	x_data = [ x*180/pi ]

	diff = 0.
	restart = False

	#random.seed( 0 )

	reward_graph = Monitor( titles='Average reward per trial', xlabel='trials', keep=False )

	with Loop_handler() as interruption :

		while not interruption() and ntrial < 2000 :

			# Action selection:
			exploration = random.rand() < prob_expl( ntrial )
			if exploration :
				u = umax*( 2*random.rand() - 1 )
			else :
				u = umax*actor.eval( scaling( x ) )
			u = clip( u, -umax, umax )

			# Simulation step:
			x_prev = array( x )
			x = odeint( model, x, [ t, t+step ], (u,), Dmodel )[-1]
			x_int = x[0]
			x[0] = ( x[0] + pi )%( 2*pi ) - pi
			diff += x_int - x[0]
			t += step

			# Computation of the reward:
			R = -cos( x[0] - pi )

			# Computation of the temporal difference:
			V_prev = critic.eval( scaling( x_prev ) )
			TD = R + gamma*critic.eval( scaling( x ) ) - V_prev

			# Update of the critic:
			#if not exploration:
			if not exploration or TD > 0 :
				critic.inc_training( scaling( x_prev ), V_prev + TD )

			# Update of the actor:
			if exploration and TD > 0 :
				actor.inc_training( scaling( x_prev ), u )

			Rt += R
			x_data.append( x*180/pi )

			if t >= Ttrial - step/2 or restart :
				ntrial += 1

				Qa = actor.end_of_batch()[0]
				Qc = critic.end_of_batch()[0]

				if sys.argv[-1] != 'quick' or ntrial%20 == 0 :
					t, Rt, success_rate, Nt = quick_eval( lambda x : umax*( clip( actor.eval( scaling( x ) ), -1, 1 ) ) )
					print( 'Eval: %i | t: %4.1f | Rt: %+7.4f | Success rate: %5.1f %% | Nt: %+3d | Qa: %7.2g | Qc: %7.2g' % ( ntrial, t, Rt, success_rate, Nt, Qa, Qc ) )
					reward_graph.add_data( ntrial, Rt )
				else :
					Rt /= t/step
					success_rate = sum( [ ( 1. if abs( a[0] ) < 10 else 0. ) for a in x_data ] )/len( x_data )*100
					print( 'Trial: %i | t: %4.1f | Rt: %+7.4f | Success rate: %5.1f %% | Nt: %+3d | Qa: %7.2g | Qc: %7.2g' % ( ntrial, t, Rt, success_rate, ( diff - x0[0] )/( 2*pi ), Qa, Qc ) )
				sys.stdout.flush()

				t = 0.
				x = array( x0 )
				Rt = 0
				x_data = [ x*180/pi ]
				diff = 0.
				restart = False

	answer = input( '\nSave data? (y) ' )
	if answer == 'y' :

		try :
			os.makedirs( data_dir )
		except OSError as e :
			if not os.path.isdir( data_dir ) :
				raise

		actor.save( data_dir + '/actor' )
		critic.save( data_dir + '/critic' )
		print( 'Data saved.' )

	#answer = input( '\nPlot the training history? (y) ' )
	#if answer == 'y' :
		#actor.plot( pause=False, name='actor' )
		#critic.plot( pause=True, name='critic' )
	
	exit( 0 )


#####################
# POLICY EVALUATION #
#####################

actor.load( data_dir + '/actor' )
critic.load( data_dir + '/critic' )

# Initial conditions:
x0 = array([ 180*pi/180, 0*pi/180 ])

tf = 10

t = 0.
x = array( x0 )

Rt = 0.
t_data = [ t ]
u_data = []
x_data = [ x*180/pi ]
theta_abs = [ x[0]*180/pi ]

diff = 0.

while t < tf - step/2 :

	# Action selection:
	#u = umax
	u = umax*actor.eval( scaling( x ) )
	u = clip( u, -umax, umax )

	# Simulation step:
	x = odeint( model, x, [ t, t+step ], (u,), Dmodel )[-1]
	x_int = x[0]
	x[0] = ( x[0] + pi )%( 2*pi ) - pi
	diff += x_int - x[0]
	t += step

	Rt += -cos( x[0] - pi )

	t_data.append( t )
	u_data.append( u )
	x_data.append( x*180/pi )
	theta_abs.append( ( diff + x[0] )*180/pi )


Rt /= tf/step + 1
success_rate = sum( [ ( 1. if abs( a[0] ) < 10 else 0. ) for a in x_data ] )/len( x_data )*100
print( 't: %4.1f | Rt: %+7.4f | Success rate: %5.1f %% | Nt: %+3d' % ( t, Rt, success_rate, ( diff - x0[0] )/( 2* pi ) ) )


############
# PLOTS 2D #
############

fig, ax = subplots( 3, sharex=True )
fig.canvas.set_window_title( 'State (' + data_id + ')' )
ax[0].set_ylabel( 'u' )
ax[0].plot( t_data[:-1], u_data )
ax[0].set_ylim( [ -1, 1 ] )
ax[0].grid( True )
ax[1].set_ylabel( u'$\\theta$' )
ax[1].plot( t_data, theta_abs )
#ax[1].plot( t_data, [ x[0] for x in x_data ] )
#ax[1].set_ylim( [ -180, 180 ] )
ax[1].grid( True )
ax[2].set_ylabel( u'$\omega$' )
ax[2].plot( t_data, [ x[1] for x in x_data ] )
ax[2].grid( True )
xlabel( u'$t$' )
xlim([ t_data[0], t_data[-1] ])


#from scipy.special import ellipk
#T = 4*sqrt(l/g)*ellipk( sin( 90*pi/360 )**2 )
#print T
#plot( time, 90*sin( 2*pi/T*time + pi/2 ) )


############
# PLOTS 3D #
############

from mpl_toolkits.mplot3d import Axes3D

resolution = 100
x_scale = linspace( -180, 180, resolution )
y_scale = linspace( -4*180, 4*180, resolution )
X, Y = meshgrid( x_scale, y_scale )
Za = zeros( ( resolution, resolution ) )
Zc = zeros( ( resolution, resolution ) )
for i, x in enumerate( x_scale ) :
	for j, y in enumerate( y_scale ) :
		Za[j][i] = umax*actor.eval( scaling( [ x*pi/180, y*pi/180 ] ) )
		Zc[j][i] = critic.eval( scaling( [ x*pi/180, y*pi/180 ] ) )

zc = []
for x in x_data :
	zc.append( critic.eval( scaling( x*pi/180 ) ) + 5 )

elev = 30 ; azim = -120

fig = figure( 'Actor (' + data_id + ')' )
ax = axes( projection='3d' )
ax.plot_surface( X, Y, Za )
#ax.set_zlim3d( -1, 1 )
ax.set_xlabel( '$\\theta$' )
ax.set_ylabel( '$\omega$' )
ax.set_zlabel( 'u' )
ax.view_init( elev, azim )

fig = figure( 'Critic (' + data_id + ')' )
ax = axes( projection='3d' )
ax.plot_surface( X, Y, Zc )
#ax.set_zlim3d( -1, 1 )
ax.set_xlabel( '$\\theta$' )
ax.set_ylabel( '$\omega$' )
ax.set_zlabel( 'V' )
ax.view_init( elev, azim )

ax.scatter( [ x[0] for x in x_data ], [ x[1] for x in x_data ], zc, c='r', marker='o', alpha=1 )


show()
