import numpy as np

def dist(x1, x2, p):
	'''
	Calculates the distance between two vectors x1 and x2.\n
 	p: int
		Power parameter for the Minkowski metric.
		e.g., p=1 for Manhattan and p=2 for Euclidian distance.
	'''

	x1_len = len(x1)
	x2_len = len(x2)

	# Make sure they are of same length.
	if x1_len == x2_len: None
	else: raise ValueError('Vectors must be of same lenght.')

	# Distances (according to the metric given) between features.
	dist_p = [np.abs(x1[i] - x2[i])**p for i in range(x1_len)]

	# The distance.
	dist_ = sum(dist_p)**(1/p)

	return dist_