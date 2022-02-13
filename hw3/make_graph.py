import pickle
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from lrc import *

lambdas = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-2, 1e-1, 1]

def 
for lmd in lambdas:
	lrc = LRC(l2_penalty=lmd)
	lrc.train(train_df
