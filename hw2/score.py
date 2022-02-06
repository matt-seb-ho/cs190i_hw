import numpy as np
import sys

pred_file = sys.argv[1]
ans_file = sys.argv[2]

pred = np.genfromtxt(pred_file, dtype='i1', delimiter=',')
ans = np.genfromtxt(ans_file, dtype='i1', delimiter=',')
total = len(ans)

perf = pred == ans
correct = np.count_nonzero(perf)
print(f'correct: {correct}, total: {total}, acc: {correct/total}')
