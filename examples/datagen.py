"""
Generate the data according to Heirarchical Dirichlet Hawkes Process
"""

import hdhp
import numpy as np
import json

def eventsToJsonFile (events, filename):
    with open (filename, "w") as fout:
        js = json.loads ("{}")
        for i, event in enumerate (events):
            js["id"] = i
            js["time"] = event[0]
            js["docs"] = event[1]
            js["users"] = event[2]
            js["meta"] = event[3]
            fout.write (json.dumps(js) + "\n")
        
def main ():
    # parameters of the model
    vocabTypes = ["docs", "auths"]
    vocabSizes = {"docs": 100, "auths": 20}
    vocabs = {vocabType: ["{0}_{1}".format (vocabType, i) for i in xrange (vocabSizes[vocabType])] for vocabType in vocabTypes}
    alpha_0 = (2.5, 0.75) # prior for excitation
    mu_0 = (2, 0.5) # prior for base intensity
    omega = 3.5 # decay kernel
    targetFile = "examples/sample_events.jsonline"

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
                              random_state=12,
                              generate=True)

    # generate events from the process
    events = process.generate (min_num_events=100, max_num_events=100000, t_max=30, reset=True)
    # events is a list with the following fields
    # t: time of the event
    # docs: documents for the event
    # users: a list of the cousers of this event
    # metadata: a list
    eventsToJsonFile (events, targetFile)
    print 'Total #events', len(events)

    with open ("examples/mu_true.tsv", "w") as fout:
        for key in process.mu_per_user:
            fout.write ("\t".join ([str (key), str (process.mu_per_user[key])]) + "\n")

    with open ("examples/alpha_true.tsv", "w") as fout:
        for key in process.time_kernels:
            fout.write ("\t".join ([str (key), str (process.time_kernels[key])]) + "\n")

    trueLabs = [e[1] for e in process.annotatedEventsIter ()]
    with open ("examples/clusters_true.txt", "w") as fout:
        for trueLab in trueLabs:
            fout.write (str (trueLab) + "\n")

if __name__ == "__main__":
    main ()
