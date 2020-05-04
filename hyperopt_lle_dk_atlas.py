from hyperopt import hp, fmin, tpe, Trials
from lle_68_base import lle
import numpy as np
import pickle

space = {
    'n_neighbors': hp.quniform('n_neighbors', 40, 10000, 20),
    'method': hp.choice('method', ['standard', 'hessian', 'modified'])
}

def optimize_lle():
    max_evals = nb_evals = 1
    try:
        trials = pickle.load(open("results.pkl", "rb"))
        print("Found saved Trials! Loading...")
        max_evals = len(trials.trials) + nb_evals
        print("Rerunning from {} trials to add another one.".format(
            len(trials.trials)))

    except:
        trials = Trials()

    best = fmin(lle, space, algo=tpe.suggest, trials=trials, max_evals=max_evals)


    pickle.dump(trials, open("results.pkl", "wb"))


if __name__ == '__main__':

    while True:
        optimize_lle()
