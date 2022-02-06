#!/usr/bin/env python
import numpy as np
import pandas as pd
# import the required packages here

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

	# read data from Xtrain_file, Ytrain_file and test_data_file
	train_data = np.genfromtxt(Xtrain_file, dtype='i', delimiter=',')
	train_labels = np.genfromtxt(Ytrain_file, dtype='i', delimiter=',')
	test_data = np.genfromtxt(test_data_file, dtype='i', delimiter=',')

	# count
	num_docs, vocab_size = np.shape(train_data)
	fake_docs = np.count_nonzero(train_labels)
	real_docs = num_docs - fake_docs

	# boolean mask to select rows
	fake_data = train_data[train_labels.astype(bool)]
	real_data = train_data[~train_labels.astype(bool)]
	
	# gather word counts per class label
	fake_total = np.sum(fake_data) + vocab_size
	fake_data = np.sum(fake_data, axis=0) + 1
	real_total = np.sum(real_data) + vocab_size
	real_data = np.sum(real_data, axis=0) + 1

	# find log likelihood vectors
	# P(w | l) = count(L=l, w=w) / count(L=l, w=*)
	ll_fake = np.log(fake_data / fake_total)
	ll_real = np.log(real_data / real_total)

	# find prior log probabilities
	prior_fake = np.log(fake_docs / num_docs)
	prior_real = np.log(real_docs / num_docs)

	# calculate probabilities of each hypothesis for each document (row)
	p_fake = np.sum(test_data * ll_fake, axis=1) + prior_fake
	p_real = np.sum(test_data * ll_real, axis=1) + prior_real
	
	# save your predictions into the file pred_file
	pred = (p_fake > p_real).astype('i1')
	np.savetxt(pred_file, pred, fmt='%d')


if __name__ == "__main__":
	Xtrain_file = '../reference/Xtrain.csv'
	Ytrain_file = '../reference/Ytrain.csv'
	test_data_file = '../reference/Xtest.csv'
	# test_data_file = '../reference/Xtrain.csv'


	pred_file = 'predictions.csv'
	run(Xtrain_file, Ytrain_file, test_data_file, pred_file)
