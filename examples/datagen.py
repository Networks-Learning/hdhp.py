"""
Generate the data according to Heirarchical Dirichlet Hawkes Process (HDHP)
"""

import hdhp
import numpy as np
from sklearn.preprocessing import normalize

def genCoAuthorsProbMatrix (nUsers):
    #A = np.zeros ((nUsers, nUsers))
    A = np.loadtxt ("authors.txt")
    # normalize the rows so that they are probabilities
    pA = normalize (A, axis=1, norm="l1")
    return pA

def main ():
    # parameters of the model
    vocabTypes = ["docs", "auths"]
    vocabSizes = {"docs": 100, "auths": 20}
    vocabs = {vocabType: ["{0}_{1}".format (vocabType, i) for i in xrange (vocabSizes[vocabType])] for vocabType in vocabTypes}
    alpha_0 = (2.5, 0.75) # prior for excitation
    mu_0 = (2, 0.5) # prior for base intensity
    omega = 3.5 # decay kernel

    # The following parameters are not really required but useful to give some
    # structure to the generated data.
    numPatterns = 10

    expectedDocLengths = {"docs": (30, 50), "auths": (1,6)}
    words_per_pattern = {"docs": 50, "auths": 5} # check again if this should be a parameter per dictionary.

    # pass only the relevant parameters to the generative process
    for_vocabs = ["docs", "auths"]

    vocab = {vocabType: vocabs[vocabType] for vocabType in for_vocabs}
    docLen = {vocabType: expectedDocLengths[vocabType] for vocabType in for_vocabs}
    wordsPerPattern = {vocabType: words_per_pattern[vocabType] for vocabType in for_vocabs}

    # generate events from the process
    numUsers = 10

    # generate a co-authorship matrix between users;
    # this could be a generative step but for now it is harcoded
    coauthors = genCoAuthorsProbMatrix (numUsers)
    print coauthors

    # create the process object
    process = hdhp.HDHProcess(num_patterns=numPatterns, 
                              alpha_0=alpha_0, 
                              mu_0=mu_0, 
                              vocabulary=vocab,
                              omega=omega,
                              doc_lengths=docLen,
                              words_per_pattern=wordsPerPattern,
                              random_state=12)
 
    events = process.generate (numUsers, coauthors, min_num_events=100, max_num_events=5000, t_max=365, reset=True)
    # Note: events is a list; each event is a quadruple -- time,docs,author,meta
    print 'Total #events', len(events)

if __name__ == "__main__":
    main ()
