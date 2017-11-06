"""
Generate the data according to Heirarchical Dirichlet Hawkes Process
"""

import hdhp
import numpy as np

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

    # create the process object
    numUsers = 10
    cousersMatrix = np.loadtxt ("examples/authors.txt")
    process = hdhp.HDHProcess(num_users = numUsers,
                              num_patterns=numPatterns, 
                              alpha_0=alpha_0, 
                              mu_0=mu_0, 
                              vocabulary=vocab,
                              omega=omega,
                              doc_lengths=docLen,
                              words_per_pattern=wordsPerPattern,
                              cousers = cousersMatrix,
                              random_state=12)

    # generate events from the process
    events = process.generate (min_num_events=100, max_num_events=None, t_max=365, reset=True)
    print 'Total #events', len(events)

if __name__ == "__main__":
    main ()
