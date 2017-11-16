import argparse
import datetime
import hdhp
import seaborn as sns; sns.set(color_codes=True) 
import pandas as pd
from collections import Counter
from sklearn.metrics import normalized_mutual_info_score
import pickle
import json

def getArgs ():
    parser = argparse.ArgumentParser ()
    parser.add_argument ("--indices", nargs="+", type=int, dest="indices", required=False, default=[0], help="indices of the vocabulary")
    args = parser.parse_args ()
    return args

def plotMuScatterPlot (xdict, ydict, outFile):
    keys = xdict.keys()
    x = pd.Series([xdict[k] for k in keys], name="exact")
    y = pd.Series([ydict[k] for k in keys], name="inferred")
    ax = sns.regplot(x=x, y=y, marker="+")
    fig = ax.get_figure ()
    fig.savefig (outFile)

def plotAlphaScatterPlot (xdict, ydict, outFile):
    xkeys = xdict.keys()
    ykeys = ydict.keys()

    inter = set (xkeys) & set (ykeys)
    x = pd.Series([xdict[k] for k in inter], name="exact")
    y = pd.Series([ydict[k] for k in inter], name="inferred")
    ax = sns.regplot (x=x, y=y, marker="+")
    fig = ax.get_figure () 
    fig.savefig (outFile)

def jsonFileToEvents (targetFile):
    events = list ()
    with open (targetFile) as fin:
        for line in fin:
            js = json.loads (line.strip())
            event = (js["time"], js["docs"], js["users"], js["meta"])
            events.append (event)

    return events

def main ():
    targetFile = "examples/sample_events.jsonline"
    types = ["docs"]
    vocabSizes = {"docs": 100}
    expectedDocLengths = {"docs": (30, 50)}
    words_per_pattern = {"docs": 50}
    vocabulary = {docType: [docType + "_{0}".format (i) for i in xrange (vocabSizes[docType])] for docType in types}

    args = getArgs ()

    # priors to control the time dynamics of the events
    alpha_0 = (2.5, 0.75)
    mu_0 = (2, 0.5)
    o = 3.5

    num_patterns = 10
    num_users = 10

    # Inference

    rawEvents = jsonFileToEvents (targetFile)
    types = [types[i] for i in args.indices]
    events = [(event[0],{t: event[1][t] for t in types}, [event[2][0]], event[3]) for event in rawEvents]

    particle, norms = hdhp.infer(events, 
                                 types,
                                 alpha_0,
                                 mu_0,
                                 omega=o,
                                 beta=1, 
                                 threads=1, 
                                 num_particles=20, 
                                 keep_alpha_history=True,
                                 seed=512)

    inf_process = particle.to_process ()

if __name__ == "__main__":
    main ()
