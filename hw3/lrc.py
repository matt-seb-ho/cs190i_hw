import numpy as np
from time import perf_counter

class LRC:
    def __init__(self, learn_rate=1e-3, l2_penalty=1e-5, 
                 converge_diff=1e-5, adagrad_eps=1e-8):
        self.learn_rate = learn_rate
        self.l2_penalty = l2_penalty
        self.converge_diff = converge_diff
        self.eps = adagrad_eps
        self.weights = None
        self.b1 = 0.9
        self.b2 = 0.999

    def train(self, data, labels, max_iter=100, report=False):
        # init weights, gradient variance (adagrad), and obj function vals
        if self.weights is None:
            self.weights = np.zeros(data.shape[1])
        # scale = np.zeros(data.shape[1])

        # adam: momentum + RMSprop 
        vdw = np.zeros(data.shape[1])
        sdw = np.zeros(data.shape[1])
        
        iters, obj, prev_obj = 0, self.converge_diff + 1, 0 

        # start timer
        start = perf_counter()

        # while objective function hasn't converged 
        while (abs(obj - prev_obj) > self.converge_diff) and iters < max_iter:
            if report and iters and iters % 30 == 0:
                acc = self.eval(data, labels, report=False)
                print(f'on iter {iters}, with obj_fn: {obj} and acc: {acc}')
            # shuffle data
            order = np.random.permutation(len(data))
            shuffled_data, shuffled_labels = data[order], labels[order]
            for example, label in zip(shuffled_data, shuffled_labels):
                p = self.p(example)
                gradient = ((label - p) * example 
                            - 2 * self.l2_penalty * self.weights)

                # attempt 1: raw gradient descent
                # self.weights += self.learn_rate * gradient

                # attempt 2: adagrad scaled adjustment
                # scale += np.square(gradient)
                # self.weights += (self.learn_rate 
                #                  * (gradient / np.sqrt(scale + self.eps)))

                # attempt 3: adam
                vdw = (self.b1 * vdw) + (1 - self.b1) * gradient
                sdw = (self.b2 * sdw) + (1 - self.b2) * np.square(gradient)
                # bias correction
                c_vdw = vdw / (1 - (self.b1 ** (iters + 1)))
                c_sdw = sdw / (1 - (self.b2 ** (iters + 1)))
                self.weights += (self.learn_rate
                                 * c_vdw
                                 / (np.sqrt(c_sdw) + self.eps))

            iters += 1
            prev_obj = obj
            obj = self.objective(data, labels)

        if report:
            print(f'finished training after {iters} iterations'
                  f'in {perf_counter() - start} seconds') 
    
    # LCL = \sum_i (y_i*log(p) + (1 - y_i)*log(1 - p))
    # objective = LCL - lambda ||w||^2
    def objective(self, data, labels):
        exponent = np.matmul(data, self.weights)
        pred = 1 / (1 + np.exp(-exponent))
        lcl = np.sum(labels * np.log(pred) + (1 - labels) * np.log(1 - pred))
        return lcl - self.l2_penalty * (np.linalg.norm(self.weights) ** 2)
                
    def p(self, example):
        exponent = np.dot(self.weights, example)
        return 1 / (1 + np.exp(-exponent))

    def eval(self, data, labels=None, pred_file=None, report=True):
        exponent = np.matmul(data, self.weights)
        predictions = 1 / (1 + np.exp(-exponent))
        predictions = (predictions > 0.5).astype('i1')
        if pred_file is not None:
            np.savetxt(pred_file, predictions, fmt='%d')

        if labels is not None:
            correct = np.sum(predictions == labels)
            n = len(labels)
            if report:
                print(f'correct: {correct}, total: {n}, accuracy: {correct/n}')
            return correct / n
