import pickle
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

report_scores = None
with open("report_scores.pickle", "rb") as f:
	report_scores = pickle.load(f)
print(report_scores)

fracs = [0.01, 0.02, 0.05, 0.10, 0.20, 1]
pcts = [1, 2, 5, 10, 20, 100]

'''
plt.plot(pcts, report_scores['nb'], label='Naive Bayes')
plt.plot(pcts, report_scores['vp'], label='Voted Perceptron')
plt.title('accuracy with various amounts of training data')
plt.xlabel('percent of the remaining training data')
plt.ylabel('accuracy')
plt.legend()
plt.show()
'''

fig, axs = plt.subplots(1, 2)
fig.suptitle('Model Accuracy with Varying Amounts of Training Data ')
axs[0].plot(pcts, report_scores['nb'])
axs[0].set_title('Naive Bayes (+Laplace Smoothing)')
axs[1].plot(pcts, report_scores['vp'])
axs[1].set_title('Voted Perceptron (100 Epochs)')
for ax in axs.flat:
	ax.set(xlabel='Percent of the Remaining Training Data', ylabel='Accuracy')

plt.show()
