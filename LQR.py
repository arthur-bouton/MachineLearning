import numpy as np
import scipy.linalg


def inverse( X ) :
	try :
		return scipy.linalg.inv( X )
	except ValueError :
		return 1/X


def lqr( A, B, Q, R ) :
	"""
	Solve the infinite-horizon and continuous-time Linear-Quadratic Regulator.

	A and B describe the continuous system dynamics:
	dx/dt = A*x + B*u

	Q and R describe the infinite-horizon quadratic-cost function minimized by the controller:
	cost = integral ( x.T*Q*x + u.T*R*u )*dt

	The optimal controller is then given by:
	u = -K*x
	"""
	Q = np.array( Q, ndmin=2 )
	R = np.array( R, ndmin=2 )

	# Solve the Ricatti equation:
	P = scipy.linalg.solve_continuous_are( A, B, Q, R )

	# Compute the LQR gain:
	K = inverse( R )@B.T@P

	# Compute the closed-loop system eigenvalues:
	eigVals, eigVecs = scipy.linalg.eig( A - B@K )

	return K, eigVals


def dlqr( A, B, Q, R ) :
	"""
	Solve the infinite-horizon and discrete-time Linear-Quadratic Regulator.

	A and B describe the discrete system dynamics:
	x[k+1] = A*x[k] + B*u[k]

	Q and R describe the infinite-horizon quadratic-cost function minimized by the controller:
	cost = sum( x[k].T*Q*x[k] + u[k].T*R*u[k] )

	The optimal controller is then given by:
	u[k] = -K*x[k]
	"""
	Q = np.array( Q, ndmin=2 )
	R = np.array( R, ndmin=2 )

	# Solve the Ricatti equation:
	P = scipy.linalg.solve_discrete_are( A, B, Q, R )

	# Compute the LQR gain:
	M = R + B.T@P@B
	K = inverse( M )@B.T@P@A

	# Compute the closed-loop system eigenvalues:
	eigVals, eigVecs = scipy.linalg.eig( A - B@K )

	return K, eigVals


def dlqr_traj( AB_tuple_list, Q, R ) :
	"""
	Solve the finite-horizon and discrete-time Linear-Quadratic Regulator for trajectory tracking.

	AB_tuple_list: [ ( A[0], B[0] ), ( A[1], B[1] ), ... ( A[N], B[N] ) ]

	A[k] and B[k] describe the discrete system dynamics along the desired trajectory x[0:N], u[0:N-1]:
	x[k+1] = A[k]*x[k] + B[k]*u[k]

	Q and R describe the finite-horizon quadratic-cost function minimized by the controller:
	cost = x[N].T*Q*x[N] + sum( x[k].T*Q*x[k] + u[k].T*R*u[k] ) for k from 0 to N-1

	The optimal controller is then given by:
	u[k] = -K[k]*x[k]
	"""

	P = np.zeros( len( AB_tuple_list ), dtype=object )
	P[-1] = Q
	for i, ( A, B ) in enumerate( reversed( AB_tuple_list[1:] ) ) :
		Pk = P[-i-1]
		P[-i-2] = Q + A.T@Pk@A - A.T@Pk@B@inverse( R + B.T@Pk@B )@B.T@Pk@A

	K = np.zeros( len( AB_tuple_list ) - 1, dtype=object )
	for i, ( A, B ) in enumerate( AB_tuple_list[:-1] ) :
		K[i] = inverse( R + B.T@P[i+1]@B )@B.T@P[i+1]@A

	return K


def discretize_system_ZOH( A, B, dt ) :
	"""
	Discretize the continuous system described by
		d/dt x( t ) = A*x( t ) + B*u( t )
		with u( t ) constant for t in [ t, t + dt ) (Zero-Order Hold)
	into a description
		x[k+1] = Ad*x[k] + Bd*u[k]
		where x[k] = x( k*dt )
	"""

	dimx = A.shape[1]
	dimu = B.shape[1]

	M = np.zeros([ dimx + dimu, dimx + dimu ])
	M[:dimx,:dimx] = A
	M[:dimx,dimx:] = B

	Md = scipy.linalg.expm( M*dt )
	Ad = Md[:dimx,:dimx]
	Bd = Md[:dimx,dimx:]

	return Ad, Bd


def discretize_system_FOH( A, B, dt ) :
	"""
	Discretize the continuous system described by
		d/dt x( t ) = A*x( t ) + B*u( t )
		with u( t + T ) = u( t ) + ( u( t + dt ) - u( t ) )*T/dt for T > 0 (First-Order Hold)
	into a description
		x[k+1] = Ad*x[k] + Bd*u[k]
		where x[k] = x( k*dt )
	"""
	from scipy.signal import cont2discrete

	Ad, Bd, Cd, Dd, dt = cont2discrete( ( A, B, np.eye( A.shape[1] ), 0 ), dt, method='foh' )

	return Ad, Bd
