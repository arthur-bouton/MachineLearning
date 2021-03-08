#!/usr/bin/env python
from looptools import Datafile
import pandas as pd
import matplotlib.pyplot as plt
from cycler import cycler
import sys


if len( sys.argv ) > 1 and sys.argv[1] == 'p' :
	env = 'pendulum'
elif len( sys.argv ) > 1 and sys.argv[1] == 'c' :
	env = 'cartpole'
else :
	print( 'Please specify the environment [p/c]', file=sys.stderr )
	exit( -1 )



file_list = []

if env == 'pendulum' :
	file_list.append( 'benchmark_pendulum_ppo.dat' )
	file_list.append( 'benchmark_pendulum_ddpg.dat' )
	file_list.append( 'benchmark_pendulum_sac.dat' )
	file_list.append( 'benchmark_pendulum_td3.dat' )
if env == 'cartpole' :
	file_list.append( 'benchmark_cartpole_ppo.dat' )
	file_list.append( 'benchmark_cartpole_ddpg.dat' )
	file_list.append( 'benchmark_cartpole_sac.dat' )
	file_list.append( 'benchmark_cartpole_td3.dat' )


R_mean = []
R_std = []
R_min = []
R_max = []
R_median = []
R_low = []
R_high = []
for f in file_list :
	df = pd.DataFrame( Datafile( f, ncols=4 ).get_data_by_rows(), columns=[ 'It', 'Ep', 'LQ', 'R' ] )
	R = df.groupby( 'Ep' ).R
	R_mean.append( R.mean() )
	R_std.append( R.std() )
	R_min.append( R.min() )
	R_max.append( R.max() )

	if env == 'pendulum' :
		p = 0.75
	if env == 'cartpole' :
		p = 0.95
	R_median.append( R.quantile( 0.5 ) )
	R_low.append( R.quantile( 1 - p ) )
	R_high.append( R.quantile( p ) )

	tdata = Datafile( f, filter='time' ).get_data()
	print( '[%s] counts: %i, average time: %f' % ( f, R.count().iloc[-1], sum( tdata )/len( tdata ) ) )



fig, ax = plt.subplots()
handles = []
for mean, std, rmin, rmax, median, rlow, rhigh in zip( R_mean, R_std, R_min, R_max, R_median, R_low, R_high ) :
	#h1 = ax.plot( mean )
	h1 = ax.plot( median )
	#h2 = ax.fill_between( mean.index, mean - std, mean + std, alpha=0.5 )
	#h2 = ax.fill_between( mean.index, rmin, rmax, alpha=0.5 )
	h2 = ax.fill_between( mean.index, rlow, rhigh, alpha=0.5 )
	handles.append( ( h1[0], h2 ) )

ax.set_xlabel( 'Number of training trials' )
ax.grid( True )
#leg = ax.legend( file_list )
#leg = ax.legend( [ 'PPO', 'DDPG', 'SAC', 'TD3' ], loc=4 )
#for line in leg.get_lines():
    #line.set_linewidth( 10 )
leg = ax.legend( handles, [ 'PPO', 'DDPG', 'SAC', 'TD3' ], loc=4 )

if env == 'pendulum' :
	ax.set_title( 'Benchmark with the pendulum problem over 200 runs' )
	ax.set_ylabel( 'Average reward per trial\n(shaded regions delineate the 25th and 75th percentiles)' )
	ax.set_xlim( [ 0, 100 ] )
if env == 'cartpole' :
	ax.set_title( 'Benchmark with the cart-pole problem over 200 runs' )
	ax.set_ylabel( 'Average reward per trial\n(shaded regions delineate the 5th and 95th percentiles)' )
	ax.set_xlim( [ 0, 600 ] )

plt.subplots_adjust( left=0.14, right=0.97, top=0.94, bottom=0.1 )



if len( sys.argv ) > 2 and sys.argv[-1] == 'w' :
	if env == 'pendulum' :
		fig.savefig( 'illustrations/benchmark_pendulum.png' )
	if env == 'cartpole' :
		fig.savefig( 'illustrations/benchmark_cartpole.png' )
else :
	plt.show()
