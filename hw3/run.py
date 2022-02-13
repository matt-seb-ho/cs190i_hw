#!/usr/bin/env python
import numpy as np
from lrc import *

def run(train_data, train_labels, test_data, pred_file):
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

	## your implementation here
	# read data from Xtrain_file, Ytrain_file and test_data_file
	train_data = np.genfromtxt(train_data, dtype='i1', delimiter=',')
	test_data = np.genfromtxt(test_data, dtype='i1', delimiter=',')
	train_labels = np.genfromtxt(train_labels, dtype='i1', delimiter=',')
	
	# your algorithm
	lrc = LRC()
	lrc.train(train_data, train_labels, max_iter=1280)

	# save your predictions into the file pred_file
	lrc.eval(test_data, pred_file=pred_file)
