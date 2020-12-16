import numpy as np

from stuff import dist
from statistics import mode

class KNN():
	'''
	KNeighboursClassifier.\n
	Parameters
	----------
	
	k: int, default=3 
		The number of neighbours.\n
	p: int, default=2
		Power parameter in Minkowski metric. 
		p=1 for Manhattan, p=2 for Euclidian, etc.
	'''

	def __init__(self, k=3, p=2):
		self.k = k
		self.p = p

	def fit(self, X, y):
		# The feature vector. Assumed to be a pd.DataFrame.
		self.X = X 
		# The label. Assumed to be a pd.Series.
		self.label = y

		return self
	
	def nearest(self, train, test):
		# The distance from the test point to each of the training points. 
		distances = [dist(test, train.iloc[i], p=self.p)
						for i in range(train.shape[0])]
		
		# Sorting the distances to get k-nearest of them.
		distances = [i for i in enumerate(distances)]
		distances.sort(key=lambda x: x[1])
		
		# Getting the indices of the nearest k points to turn them into labels.
		nearest_indices = [distances[i][0] for i in range(self.k)]
		
		# Get the labels.
		labels = [self.label.iloc[i] for i in nearest_indices]
		# Get the most frequent label. This is like voting.
		label = mode(labels)
		
		return label



	def predict(self, X):
		# Test set, assumed to be a pd.DataFrame.
		test = X

		labels = [self.nearest(train=self.X, test=test.iloc[i]) 
					for i in range(test.shape[0])]

		return np.array(labels)