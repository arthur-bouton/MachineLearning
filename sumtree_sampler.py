"""
Author: Arthur Bouton [arthur.bouton@gadz.org]

"""
from numpy import zeros
import random


class Sumtree_sampler :
	"""
	A sum tree structure to efficiently sample items according to their relative priorities.

	Parameter
	---------
	capacity : int
		The maximum amount of items that will be possibly stored.

	"""

	def __init__( self, capacity ) :
		self.capacity = capacity
		self.tree = zeros( 2*capacity - 1 )
		self.data = zeros( capacity, dtype=object )
		self.next_index = 0
		self.full = False
		self.max_p_seen = 1

	def __len__( self ) :
		""" Get the current number of items stored so far """

		if not self.full :
			return self.next_index
		else :
			return self.capacity

	def _propagate( self, leaf_index, change ) :

		parent_index = ( leaf_index - 1 )//2

		self.tree[parent_index] += change

		if parent_index > 0 :
			self._propagate( parent_index, change )

	def _retrieve( self, value, leaf_index=0 ) :

		left = 2*leaf_index + 1
		right = left + 1

		if left >= len( self.tree ) :
			return leaf_index

		if value < self.tree[left] or self.tree[right] == 0 :
			return self._retrieve( value, left )
		else :
			return self._retrieve( value - self.tree[left], right )

	def append( self, data, p=None ) :
		""" Add a new item with priority p """

		if p is None :
			p = self.max_p_seen
		else :
			self.max_p_seen = max( self.max_p_seen, p )

		self.data[self.next_index] = data
		self.update( self.next_index, p )

		self.next_index += 1
		if self.next_index >= self.capacity :
			self.next_index = 0
			self.full = True

	def update( self, index, p ) :
		""" Update item's priority by referring to its index """

		leaf_index = index + self.capacity - 1

		self._propagate( leaf_index, p - self.tree[leaf_index] )
		self.tree[leaf_index] = p

		self.max_p_seen = max( self.max_p_seen, p )

	def sum( self ) :
		""" The total sum of the priorities from all the items stored """

		return self.tree[0]

	def sample( self, length=1 ) :
		""" Sample a list of items according to their priorities """

		data, indices, priorities = [], [], []
		for _ in range( length ) :
			leaf_index = self._retrieve( random.uniform( 0, self.tree[0] ) )
			index = leaf_index - self.capacity + 1
			data.append( self.data[index] )
			indices.append( index )
			priorities.append( self.tree[leaf_index] )

		return data, indices, priorities
