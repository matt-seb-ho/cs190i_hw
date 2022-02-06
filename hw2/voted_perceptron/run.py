#!/usr/bin/env python
# import the required packages here
import numpy as np
from time import perf_counter

class VotedPerceptron:
	def __init__(self):
		self.weights = []
		self.correct = []

	def train(self, feature_file, label_file, rounds, report_time=False):
		start = perf_counter()
		features = np.genfromtxt(feature_file, dtype='i', delimiter=',')
		train_size, feature_size = features.shape
		labels = np.genfromtxt(label_file, dtype='i', delimiter=',')

		w = np.zeros(feature_size)
		self.weights.append(w)
		self.correct.append(0)
		for _ in range(rounds):
			for idx, point in enumerate(features):
				# convert [0,1] => [-1, 1]
				label = [-1, 1][labels[idx]]
				if label * np.dot(w, point) <= 0:
					w = np.copy(w) + label * features[idx]
					self.weights.append(w)
					self.correct.append(1)
				else:
					self.correct[-1] += 1

		# convert to numpy arrays
		self.weights = np.stack(self.weights)
		self.correct = np.array(self.correct)
		if report_time:
			print(f'trained for {rounds} rounds in {perf_counter() - start}s')

	def pred(self, feature_file, pred_file):
		features = np.genfromtxt(feature_file, dtype='i', delimiter=',')
		features = np.transpose(features)
		
		# dot products; res[i][j] = perceptron i's dot product with point j
		res = np.matmul(self.weights, features)
		res = np.sign(res)
		# scaling each perceptron's prediction by its training performance
		res *= np.transpose([self.correct])
		# sum each column and get sign as prediction
		res = np.sign(np.sum(res, axis=0))
		
		# replace -1s with 0s
		res = np.sign(res + 1)
		np.savetxt(pred_file, res, fmt="%d")


def run(Xtrain_file, Ytrain_file, test_data_file, pred_file):
	'''The function to run your ML algorithm on given datasets, generate the predictions and save them into the provided file path
	
	Parameters
	----------
	Xtrain_file: string
		the path to Xtrain csv file
	Ytrain_file: string
		the path to Ytrain csv file
	test_data_file: string
		the path to test data csv file
	pred_file: string
		the prediction file to be saved by your code. You have to save your predictions into this file path following the same format of Ytrain_file
	'''
	vp = VotedPerceptron()
	vp.train(Xtrain_file, Ytrain_file, 200)
	vp.pred(test_data_file, pred_file)

if __name__ == "__main__":
	Xtrain_file = '../reference/Xtrain.csv'
	Ytrain_file = '../reference/Ytrain.csv'
	test_data_file = '../reference/Xtest.csv'
	# test_data_file = '../reference/Xtrain.csv'

	pred_file = 'predictions.csv'
	run(Xtrain_file, Ytrain_file, test_data_file, pred_file)
