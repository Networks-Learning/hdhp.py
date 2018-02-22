import matplotlib
matplotlib.use('Agg')

import argparse
import datetime
import hdhp
import seaborn as sns; sns.set(color_codes=True) 
import pandas as pd
from collections import Counter
from sklearn.metrics import normalized_mutual_info_score
import pickle
import json
import numpy as np
import os
import timeit

def getArgs ():
    parser = argparse.ArgumentParser ()
    parser.add_argument ("--indices", nargs="+", type=int, dest="indices", required=False, default=[0], help="indices of the vocabulary")
    parser.add_argument ("--use-cousers", dest="use_cousers", action="store_true")
    parser.set_defaults (use_cousers=False)
    args = parser.parse_args ()
    return args

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

def jsonFileToEvents (targetFile):
    events = list ()
    with open (targetFile) as fin:
        for line in fin:
            js = json.loads (line.strip())
            event = (js["time"], js["docs"], js["users"], js["meta"])
            events.append (event)

    return events

def plotMuScatterPlot (xdict, ydict, outFile):

    keys = xdict.keys()

    x = pd.Series([xdict[k] for k in keys], name="True Value")
    y = pd.Series([ydict[k] for k in keys], name="Inferred Value")

    max_axis_value = max(max(x), max(y)) + 1
    min_axis_value = min(min(x), min(y)) - 1

    z = np.linspace(int(min_axis_value), 10)
    sns.plt.plot(z, z + 0, linestyle='solid')

    # ax = sns.regplot(x=x, y=x)
    ax = sns.regplot(x=x, y=y, marker="+", fit_reg=False)
    ax.set(title="Fig Title: Base Intensity")

    sns.plt.ylim(min_axis_value, 10)
    sns.plt.xlim(min_axis_value, 10)
    

    fig = ax.get_figure ()
    fig.savefig (outFile)
    fig.clf()

def plotAlphaScatterPlot (xdict, ydict, outFile):
    
    xkeys = xdict.keys()
    ykeys = ydict.keys()

    inter = set (xkeys) & set (ykeys)
    x = pd.Series([xdict[k] for k in inter], name="True Value")
    y = pd.Series([ydict[k] for k in inter], name="Inferred Value")

    max_axis_value = max(max(x), max(y)) + 1
    min_axis_value = min(min(x), min(y)) - 1

    z = np.linspace(int(min_axis_value), int(max_axis_value))
    sns.plt.plot(z, z + 0, linestyle='solid')

    ax = sns.regplot (x=x, y=y, marker="o", fit_reg=False)
    ax.set(title="Fig Title: Kernel Parameter")

    sns.plt.ylim(min_axis_value, max_axis_value)
    sns.plt.xlim(min_axis_value, max_axis_value)

    fig = ax.get_figure () 
    fig.savefig (outFile)
    fig.clf()


def createTriDiagonalMatrix (n): # TODO: Think more about this distribution! ---> Symetric????
    ones = np.ones (n)
    cousers = 0 * np.diag(ones)

    for i in range(n):
        probs = np.random.dirichlet(np.ones(n - 1),size=1)
        probs = np.insert(probs, i, 1)
        cousers[i, :] = probs
    return cousers

def generate (num_users, num_events, num_patterns, vocab_sizes, time_horizon, alpha_0, mu_0, omega, expected_doc_lengths, words_per_pattern):
    # parameters of the model
    vocabTypes = ["docs", "auths"]
    vocabs = {vocabType: ["{0}_{1}".format (vocabType, i) for i in xrange (vocab_sizes[vocabType])] for vocabType in vocabTypes}
    targetFile = "examples/sample_events.jsonline"

    # pass only the relevant parameters to the generative process
    for_vocabs = ["docs", "auths"]

    vocab = {vocabType: vocabs[vocabType] for vocabType in for_vocabs}
    docLen = {vocabType: expected_doc_lengths[vocabType] for vocabType in for_vocabs}
    wordsPerPattern = {vocabType: words_per_pattern[vocabType] for vocabType in for_vocabs}

    # create the process object
    cousersMatrix = createTriDiagonalMatrix (num_users)
    # print("cusers: " + str(cousersMatrix))

    process = hdhp.HDHProcess(num_users = num_users,
                              num_patterns=num_patterns, 
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
    events = process.generate (min_num_events=20, max_num_events=num_events, t_max=time_horizon, reset=True)

    # events is a list with the following fields
    # t: time of the event
    # docs: documents for the event
    # users: a list of the cousers of this event
    # metadata: a list
    eventsToJsonFile (events, targetFile)
    print 'Total #events', len(events)
    return process

def infer (num_users, num_patterns, alpha_0, mu_0, omega, num_particles, indices, use_cousers=False):
    targetFile = "examples/sample_events.jsonline"
    types = ["docs", "auths"]
    # priors to control the time dynamics of the events

    # alpha_0 = (4.0, 0.5) # prior for excitation
    # mu_0 = (8, 0.25) # prior for base intensity
    # o = 1.5 # decay kernel

    # Inference
    rawEvents = jsonFileToEvents (targetFile)
    types = [types[i] for i in indices]
    events = list ()
    if use_cousers:
        for event in rawEvents:
            events.append ((event[0], {t: event[1][t] for t in types}, event[2], event[3]))
    else:
        for event in rawEvents:
            events.append ((event[0], {t: event[1][t] for t in types}, [event[2][0]], event[3]))

    particle, norms = hdhp.infer(events,
                                 num_users,
                                 types,
                                 alpha_0,
                                 mu_0,
                                 omega=omega,
                                 beta=1, 
                                 threads=1,
                                 num_particles=num_particles, 
                                 keep_alpha_history=True,
                                 seed=512)

    inf_process = particle.to_process ()
    return inf_process

def main ():
    #args = getArgs ()
    # Parameters to generate the data
    num_users = 200
    num_events = 150000
    num_patterns = 10
    vocab_sizes = {"docs": 100, "auths": 100}
    time_horizon = 3000
    alpha_0 = (8, 0.25) # prior for excitation
    mu_0 = (10, 0.2) # prior for base intensity
    omega = 5 # decay kernel
    expected_doc_lengths = {"docs": (100, 150), "auths": (20,25)}
    words_per_pattern = {"docs": 30, "auths": 30} # check again if this should be a parameter per dictionary.
    # generate the data

    print("Generated Info:  ")
    print("Number of Users: " + str(num_users))
    print("Number of Events: " + str(num_events))
    print("Number of Patterns: " + str(num_patterns))

    genHDHP = generate (num_users, num_events, num_patterns, vocab_sizes, time_horizon, alpha_0, mu_0, omega, expected_doc_lengths, words_per_pattern)

    print("Generation is done!")

    num_particles = 20

    cases = {1: ([0], False),
             2: ([0,1], False),
             3: ([0,1], True)}

    for case in [1, 2, 3]:

        print "Case: {0}".format (case)
        start = timeit.default_timer()

        indices, use_cousers = cases[case]
        if not os.path.exists ("results/{0}".format (case)):
            os.makedirs ("results/{0}".format (case))

        dirname = "results/{0}".format (case)
        # infer the parameters from the data
        infHDHP = infer (num_users, num_patterns, alpha_0, mu_0, omega, num_particles, indices, use_cousers)
        print("Inference is done in " + str(timeit.default_timer() - start) + " seconds!")


        with open (os.path.join (dirname, "U_" + str(num_users) + "_E_" + str(num_events) + "_base_rates.tsv"), "w") as fout:
            for key in genHDHP.mu_per_user:
                fout.write ("\t".join ([str (key), str (genHDHP.mu_per_user[key]), str (infHDHP.mu_per_user[key])]) + "\n")

        with open (os.path.join (dirname, "U_" + str(num_users) + "_E_" + str(num_events) + "_set_time_kernels.tsv"), "w") as fout:
            for key in genHDHP.time_kernels:
                fout.write ("\t".join ([str (key), str (genHDHP.time_kernels[key])]) + "\n")

        with open (os.path.join (dirname, "U_" + str(num_users) + "_E_" + str(num_events) + "_est_time_kernels.tsv"), "w") as fout:
            for key in infHDHP.time_kernels:
                fout.write ("\t".join ([str (key), str (infHDHP.time_kernels[key])]) + "\n")
   

        # plot the base rates and the estimated alpha values
        plotMuScatterPlot (genHDHP.mu_per_user, infHDHP.mu_per_user, "figs/" + "Case:{0}".format(case) + "_U_" + str(num_users) + "_E_" + str(num_events) + "_base_rates.pdf")
        plotAlphaScatterPlot (genHDHP.time_kernels, infHDHP.time_kernels, "figs/" + "Case:{0}".format(case) + "_U_" + str(num_users) + "_E_" + str(num_events) + "_time_kernels.pdf")
        
        trueLabs = [e[1] for e in genHDHP.annotatedEventsIter ()]
        predLabs = [e[1] for e in infHDHP.annotatedEventsIter ()]

        print("True Labels Size: " + str(len(trueLabs)))
        print("predected Lables Size: " + str(len(predLabs)))


        with open (os.path.join (dirname, "U_" + str(num_users) + "_E_" + str(num_events) + "_patterns.tsv"), "w") as fout:
            for i in xrange (len (trueLabs)):
                fout.write ("\t".join ([str(trueLabs[i]), str (predLabs[i])]) + "\n")

        print ("NMI = " + str(normalized_mutual_info_score (trueLabs, predLabs)))


if __name__ == "__main__":
    main ()