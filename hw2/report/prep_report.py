import pickle
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from time import perf_counter

'''
Instructions
First, use the last 10% of the training data as your test data. Compare Naive Bayes and Voted Perceptron on several fractions of your remaining training data. For this purpose, pick 1%,2%,5%,10%,20% and 100% of the first 90% training data to train and compare the perfor- mance of Naive Bayes and Voted Perceptron on the test data respectively. Plot the accuracy as a function of the size of the fraction you picked (x-axis should be “percent of the remaining training data” and y-axis should be “accuracy”).
'''

def nb_acc(train_data, train_labels, test_data, test_labels):
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
	acc = np.sum(pred == test_labels) / len(test_labels)
	return acc
	

class VotedPerceptron:
	def __init__(self):
		self.weights = []
		self.correct = []

	# def train(self, feature_file, label_file, rounds, report_time=False):
	def train(self, features, labels, rounds, report_time=False):
		start = perf_counter()
		# features = np.genfromtxt(feature_file, dtype='i', delimiter=',')
		train_size, feature_size = features.shape
		# labels = np.genfromtxt(label_file, dtype='i', delimiter=',')

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

	def test_acc(self, features, labels):
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

		acc = np.sum(res == labels) / len(labels)
		return acc

# split data
data_file = '../reference/Xtrain.csv'
label_file = '../reference/Ytrain.csv'

data = np.genfromtxt(data_file, dtype='i', delimiter=',')
labels = np.genfromtxt(label_file, dtype='i', delimiter=',') 

num_samples = len(labels)
split = int(.9 * num_samples)
print(f'num_samples:{num_samples}, split:{split}')

train_data = data[:split]
test_data = data[split:]
print(f'num_train:{len(train_data)}, num_test:{len(test_data)}')
train_labels = labels[:split]
test_labels = labels[split:]

def get_train_subset(pct):
	amt = int(split * pct)
	print(f'pct:{pct}, amt:{amt}')
	return train_data[:amt], train_labels[:amt]

# NB signature
# def nb_acc(train_data, train_labels, test_data, test_labels):
def get_nb_acc(pct_train):
	td, tl = get_train_subset(pct_train)
	return nb_acc(td, tl, test_data, test_labels)

def get_vp_acc(pct_train):
	td, tl = get_train_subset(pct_train)
	vp = VotedPerceptron()
	vp.train(td, tl, 100)
	return vp.test_acc(test_data, test_labels)

pcts = [0.01, 0.02, 0.05, 0.10, 0.20, 1]
acc_scores =  {
	'pct': pcts,
	'nb': [get_nb_acc(pct) for pct in pcts],
	'vp': [get_vp_acc(pct) for pct in pcts]
}

print(acc_scores)

with open('report_scores2.pickle', 'wb') as f:
	pickle.dump(acc_scores, f)
