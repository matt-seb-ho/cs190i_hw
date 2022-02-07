import numpy as np
import pandas as pd

class LRC:
    def __init__(self, learn_rate, l2_penalty=0.2, converge_diff=1e-3):
        self.learn_rate = learn_rate
        self.l2_penalty = l2_penalty
        self.converge_diff = converge_diff
        self.weights = None
        self.eps = 1e-6

    def train(self, data, labels):
        # init weights, and objective function
        weights = np.zeros(data.shape[1])
        
        # adagrad accumulates gradient variance
        scale = np.zeros(data.shape[1])
        
        iters, obj, prev_obj = 0, None, None
        # while objective function hasn't curved
        while iters < 2 or (abs(obj - prev_obj) > converge_diff):
            order = numpy.random.permutation(len(data))
            shuffled_data, shuffled_labels = data[order], labels[order]
            for idx, example in enumerate(shuffled_data):
                prob = p(example)
                gradient = (label[idx] - prob) * example - (2 * l2_penalty) * weights
                scale += np.squared(gradient)
                weights += learn_rate * (gradient / np.sqrt(scale + self.eps))
            iters += 1
            prev_obj = obj
            obj = objective(data, labels)
    
    # LCL = \sum_i (y_i*log(p) + (1 - y_i)*log(1 - p))
    # objective = LCL - lambda ||w||^2
    def objective(self, data, labels):
        exponent = np.matmul(data, np.transpose([self.weights]))
        exponent = np.transpose(exponent)[0]
        predictions = 1 / (1 + np.exp(exponent))
        lcl = label * np.log(predictions) + (1 - label) * np.log(1 - predictions)
        return np.sum(lcl) - (self.l2_penalty * (np.linalg.norm(self.weights) ** 2))
                
    def p(example):
        exponent = -np.dot(self.weights, example)
        return 1 / (1 + np.exp(exponent))

    def eval(data, labels):
        exponent = np.matmul(data, np.transpose([self.weights]))
        exponent = np.transpose(exponent)[0]
        predictions = 1 / (1 + np.exp(exponent))
        predictions = (predictions > 0.5).astype('i1')
        correct = np.sum(predictions == labels)
        n = len(labels)
        print(f'correct: {correct}, total: {n}, accuracy: {correct / n}')
        return correct / n
