from lrc import *
from tqdm import tqdm

train_iters = 100
def kfold_cv(k, lrc, data, labels, early_stop=None):
    # setup
    sp_len = len(labels) // k
    fold_acc = np.zeros(k)
    if early_stop is None:
        early_stop = k

    for i in range(early_stop):
        # prep data
        exc_start, exc_stop = sp_len * i, sp_len * (i + 1)
        tr_data = np.delete(data, range(exc_start, exc_stop), axis=0)
        tr_labels = np.delete(labels, range(exc_start, exc_stop), axis=0)

        te_data = data[range(exc_start, exc_stop)]
        te_labels = labels[range(exc_start, exc_stop)]

        # evaluate 
        lrc.weights = None
        lrc.train(tr_data, tr_labels, max_iter=train_iters)
        fold_acc[i] = lrc.eval(te_data, te_labels, report=False)

    return fold_acc

def search_lmd(lmds, data, labels):
    res = {}
    for lmd in lmds:
        lrc = LRC(l2_penalty=lmd)
        tmp = kfold_cv(10, lrc, data, labels)
        res[lmd] = np.mean(tmp)
    return res

# grid search for best learn_rate + l2 penalty

# -------------------------------------
# experiment 1:
# alphs = [1, 1e-1, 1e-2, 1e-3, 1e-4]
# lmbds = [1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
#
# trends: increase with smaller lambda, increase 1st half alph, decrease 2nd half alph
# best overall: use a=1e-2, l=1e-5

# -------------------------------------
# experiment 2:
# alphs = [8e-3, 1e-2, 2e-2, 3e-2, 5e-2]
# lmbds = [1e-4, 5e-5, 2e-5,  1e-5, 8e-6]

# -------------------------------------
# experiment 3: prep scores for graph
alphs = [2e-2]
lmbds = [1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 0]

def grid_search(alps, lmbs, data, labels, fname):
    res = np.zeros((len(alps), len(lmbs)))
    for ai, alp in enumerate(tqdm(alps)):
        for li, lmd in enumerate(tqdm(lmbs, leave=False)):
            lrc = LRC(learn_rate=alp, l2_penalty=lmd)
            res[ai, li] = np.mean(kfold_cv(10, lrc, data, labels))
    np.save(fname, res)
    return res
