import numpy as np


class Lobatto_quad :

	def __init__( self, degree ) :
		""" Gauss-Lobatto quadrature """

		if not degree in self.lobatto_roots :
			raise ValueError( 'Degree %i not implemented' % degree )

		# Gauss–Lobatto roots:
		half_roots = self.lobatto_roots[degree]
		self.tq = np.array( list( reversed( -np.array( half_roots[(degree+1)%2:] ) ) ) + half_roots )

		# Interpolation weights:
		self.vq = np.array( [ 1/np.prod( [ ti - tj for tj in self.tq[self.tq!=ti] ] ) for ti in self.tq ] )

		# Quadrature weights:
		unscaled_weights = self.vq**2
		self.wq = unscaled_weights/sum( unscaled_weights )
	
	def _set_interval( self, xa, xb ) :

		# Interval-remapping functions:
		self.map_xt = lambda x: 2*( x - xa )/( xb - xa ) - 1
		self.map_tx = lambda t: ( t + 1 )*( xb - xa )/2 + xa

		self.delta_x = xb - xa
	
	def set_function( self, f, xa, xb ) :
		
		self._set_interval( xa, xb )

		# Sampled function evaluations:
		self.fq = np.array( [ f( self.map_tx( t ) ) for t in self.tq ] )

		return self
	
	def set_points( self, f_list, xa, xb ) :

		if len( f_list ) != len( self.tq ) :
			raise ValueError( 'Expected %i points but received %i' % ( len( self.tq ), len( f_list ) ) )
		
		self._set_interval( xa, xb )

		self.fq = np.array( f_list )

		return self
	
	def integral( self ) :
		return float( self.wq@self.fq*self.delta_x )
	
	def integral2( self ) :
		return float( self.wq@self.fq**2*self.delta_x )

	def interpolation( self, x, points=None ) :
		""" Barycentric Lagrange interpolation """

		if points is None :
			points = self.fq

		t = self.map_xt( x )
		if t in self.tq :
			return float( points[self.tq==t] )
		cq = np.array( [ vi/( t - ti ) for vi, ti in zip( self.vq, self.tq ) ] )
		return float( cq@points/sum( cq ) )

	def derivative( self, x ) :
		"""
		Compute the derivative of the interpolating polynomial using barycentric Lagrange interpolation [1].

		[1] Berrut, Jean-Paul, and Lloyd N. Trefethen. "Barycentric lagrange interpolation." SIAM review 46.3 (2004): 501-517.

		"""

		if not hasattr( self, 'dfq' ) :
			# Differentiation matrix:
			n = len( self.fq )
			D = np.zeros(( n, n ))
			for i in range( n ) :
				for j in range( n ) :
					if i != j :
						D[i][j] = self.vq[j]/self.vq[i]/( self.tq[i] - self.tq[j] )
			D -= np.diag( sum( D.T ) )

			# Differentiation of the polynomial interpolants:
			self.dfq = D@self.fq*2/self.delta_x

		return float( self.interpolation( x, self.dfq ) )

	def derivative2( self, x ) :
		"""
		Compute the second derivative of the interpolating polynomial using barycentric Lagrange interpolation [1].

		[1] Berrut, Jean-Paul, and Lloyd N. Trefethen. "Barycentric lagrange interpolation." SIAM review 46.3 (2004): 501-517.

		"""

		if not hasattr( self, 'dfq2' ) :
			# Differentiation matrix:
			n = len( self.fq )
			D2 = np.zeros(( n, n ))
			for i in range( n ) :
				for j in range( n ) :
					if i != j :
						#D2[i][j] = -2*self.vq[j]/self.vq[i]/( self.tq[i] - self.tq[j] )*( sum( self.vq[k]/self.vq[i]/( self.tq[i] - self.tq[k] ) for k in range( n ) if k != i ) - 1/( self.tq[i] - self.tq[j] ) )
						D2[i][j] = -2*self.vq[j]/self.vq[i]/( self.tq[i] - self.tq[j] )*sum( self.vq[k]/self.vq[i]/( self.tq[i] - self.tq[k] ) - 1/( self.tq[i] - self.tq[j] ) for k in range( n ) if k != i )
			D2 -= np.diag( sum( D2.T ) )

			# Differentiation of the polynomial interpolants:
			self.dfq2 = D2@self.fq
			#self.dfq2 = D2@self.fq*2/self.delta_x

		return float( self.interpolation( x, self.dfq2 ) )

	lobatto_roots = {
		1: [
				1
			],
		2: [
				0,
				1
			],
		3: [
				np.sqrt( 1/5 ),
				1
			],
		4: [
				0,
				np.sqrt( 3/7 ),
				1
			],
		5: [
				np.sqrt( 1/3 - 2*np.sqrt( 7 )/21 ),
				np.sqrt( 1/3 + 2*np.sqrt( 7 )/21 ),
				1
			],
		6: [
				0,
				np.sqrt( 5/11 - 2*np.sqrt( 5/3 )/11 ),
				np.sqrt( 5/11 + 2*np.sqrt( 5/3 )/11 ),
				1
			],
		7: [
				0.2092992179024788687687,
				0.5917001814331423021445,
				0.8717401485096066153375,
				1
			],
		8: [
				0,
				0.3631174638261781587108,
				0.6771862795107377534459,
				0.8997579954114601573124,
				1
			],
		9: [
				0.1652789576663870246262,
				0.4779249498104444956612,
				0.7387738651055050750031,
				0.9195339081664588138289,
				1
			],
		10: [
				0,
				0.2957581355869393914319,
				0.565235326996205006471,
				0.7844834736631444186224,
				0.9340014304080591343323,
				1
			],
		11: [
				0.1365529328549275548641,
				0.3995309409653489322643,
				0.6328761530318606776624,
				0.8192793216440066783486,
				0.9448992722228822234076,
				1
			],
		12: [
				0,
				0.2492869301062399925687,
				0.4829098210913362017469,
				0.6861884690817574260728,
				0.8463475646518723168659,
				0.9533098466421639118969,
				1
			],
		13: [
				0.1163318688837038676588,
				0.3427240133427128450439,
				0.5506394029286470553166,
				0.7288685990913261405847,
				0.8678010538303472510002,
				0.9599350452672609013551,
				1
			],
		14: [
				0,
				0.2153539553637942382257,
				0.4206380547136724809219,
				0.6062532054698457111235,
				0.7635196899518152007041,
				0.8850820442229762988254,
				0.9652459265038385727959,
				1
			],
		15: [
				0.101326273521949447843,
				0.2998304689007632080984,
				0.4860594218871376117819,
				0.6523887028824930894679,
				0.7920082918618150639311,
				0.8992005330934720929946,
				0.9695680462702179329522,
				1
			],
		16: [
				0,
				0.1895119735183173883043,
				0.3721744335654770419072,
				0.5413853993301015391237,
				0.6910289806276847053949,
				0.8156962512217703071068,
				0.9108799959155735956238,
				0.973132176631418314157,
				1
			],
		17: [
				0.08974909348465211102265,
				0.2663626528782809841677,
				0.4344150369121239753423,
				0.5885048343186617611735,
				0.7236793292832426813062,
				0.8355935352180902137137,
				0.9206491853475338738379,
				0.9761055574121985428645,
				1
			],
		18: [
				0,
				0.1691860234092815713752,
				0.333504847824498610299,
				0.488229285680713502778,
				0.6289081372652204977668,
				0.7514942025526130141636,
				0.852460577796646093086,
				0.9289015281525862437179,
				0.9786117662220800951526,
				1
			],
		19: [
				0.0805459372388218379759,
				0.239551705922986495182,
				0.3923531837139092993865,
				0.5349928640318862616481,
				0.6637764022903112898464,
				0.7753682609520558704143,
				0.8668779780899501413099,
				0.9359344988126654357162,
				0.9807437048939141719255,
				1
			],
		20: [
				0,
				0.1527855158021854660064,
				0.3019898565087648872754,
				0.444115783279002101195,
				0.575831960261830686927,
				0.6940510260622232326273,
				0.7960019260777124047443,
				0.8792947553235904644512,
				0.9419762969597455342961,
				0.9825722966045480282345,
				1
			]
		}
