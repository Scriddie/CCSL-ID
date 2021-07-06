import numpy as np
from scipy.stats import shapiro


def test_marginal(X):
    """ 
    Test marginal distribution assumption
    X: observational data 
    """
    n, d = X.shape
    marg_stats = []
    for i in range(d):
        marg_stats.append(shapiro(X[:, i]))
    correct = [i > 0.05 for i in marg_stats]
    if sum(correct) == 0:
        print("No correctly specified root cause found.")


def test_score_equivalence(X, score, tol=1e-2):
    """ 
    test pairwise score equivalence 
    X: observational data
    """
    n, d = X.shape
    score_mismatch = []
    modelA = np.triu(np.ones((2, 2)))
    modelB = np.tril(np.ones((2, 2)))
    for i in range(d):
        for j in range(d):
            if i == j:
                continue
            scoreA = score(X[:, [i, j]], modelA)
            scoreB = score(X[:, [i, j]], modelB)
            score_eq.append(np.abs(scoreA-scoreB)>tol)
    violations = sum(score_mismatch)
    if violations > 0:
        print(f'{violations} pairwise score equivalence violations found.')


def test_multiple_identifiability(X, interventions):
    """ 
    Test pairwise multiple identifiability
    X:  obs and int data
    """
    intervened_upon = np.unique(interventions)
    for i in intervened_upon:
        for j in intervened_upon:
            if i == j:
                continue
            