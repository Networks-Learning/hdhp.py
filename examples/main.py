import matplotlib
matplotlib.use('Agg')

import notebook_helpers

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
import itertools
import operator
import NMI
import matplotlib.pyplot as plt


def find_kernel_mapping(true_labels, estimated_labels):

    mappings = {}

    for i in range(len(true_labels)):
        true_label = true_labels[i]
        pred_label = estimated_labels[i]

        if true_label in mappings:
            if pred_label in mappings[true_label]:
                mappings[true_label][pred_label] += 1
            else:
                mappings[true_label][pred_label] = 1
        else:
            mappings[true_label] = {pred_label: 1}

    best_mapping = {}

    for true_label in mappings:
        best_mapping[true_label] = max(mappings[true_label].iteritems(), key=operator.itemgetter(1))[0]

    return best_mapping

def plotMuScatterPlot (xdict, ydict, outFile):

    keys = xdict.keys()

    x = pd.Series([xdict[k] for k in keys], name="True Value")
    y = pd.Series([ydict[k] for k in keys], name="Inferred Value")

    max_axis_value = max(max(x), max(y)) + 1
    min_axis_value = min(min(x), min(y)) - 1

    z = np.linspace(int(min_axis_value), 6)
    sns.plt.plot(z, z + 0, linestyle='solid')

    # ax = sns.regplot(x=x, y=x)
    ax = sns.regplot(x=x, y=y, marker="+", fit_reg=False)
    ax.set(title="Fig Title: Base Intensity")

    sns.plt.ylim(min_axis_value, 6)
    sns.plt.xlim(min_axis_value, 6)


    fig = ax.get_figure ()
    fig.savefig (outFile)
    fig.clf()
    plt.close(fig)

def plotAlphaScatterPlot (xdict, ydict, outFile):

    xkeys = xdict.keys()
    ykeys = ydict.keys()

    inter = set (xkeys) & set (ykeys)

    x = pd.Series([xdict[k] for k in inter], name="True Value")
    y = pd.Series([ydict[k] for k in inter], name="Inferred Value")

    max_axis_value = max(max(x), max(y)) + 1
    min_axis_value = min(min(x), min(y)) - 1

    z = np.linspace(int(min_axis_value), 6)
    sns.plt.plot(z, z + 0, linestyle='solid')

    ax = sns.regplot (x=x, y=y, marker="o", fit_reg=False)
    ax.set(title="Fig Title: Kernel Parameter")

    sns.plt.ylim(min_axis_value, 6)
    sns.plt.xlim(min_axis_value, 6)

    # sns.plt.ylim(min_axis_value, max_axis_value)
    # sns.plt.xlim(min_axis_value, max_axis_value)

    fig = ax.get_figure ()
    fig.savefig (outFile)
    fig.clf()
    plt.close(fig)


def generate(num_users, num_patterns, alpha_0, mu_0, omega, vocab_size, doc_min_length, doc_length, words_per_pattern, num_samples, vocab_types):

    vocabulary = {}
    for vocab_type in vocab_types:
        vocabulary[vocab_type] = [vocab_type + str(i) for i in range(vocab_size[vocab_type])]  # the `words` of our documents

    process = hdhp.HDHProcess(num_patterns=num_patterns, alpha_0=alpha_0, num_users=num_users, vocab_types=vocab_types,
                              mu_0=mu_0, vocabulary=vocabulary, doc_length=doc_length, doc_min_length=doc_min_length,
                              omega=omega, words_per_pattern=words_per_pattern,
                              random_state=12, generate=True)

    # overlap = notebook_helpers.compute_pattern_overlap(process)
    # ax = sns.distplot(overlap, kde=True, norm_hist=True, axlabel='Content overlap')
    # fig = ax.get_figure()
    # fig.savefig("Figs/" + str(num_patterns) + "_pattern_overlaps.pdf")
    # fig.clf()
    # plt.close(fig)

    process.reset()  # removes any previously generated data
    process.sample_user_events(min_num_events=100,
                                       max_num_events=num_samples,
                                       t_max=365)
    for cluster in process.dish_counters:
        print("Cluster " + str(cluster) + " : " + str(process.dish_counters[cluster]))

    num_events = len(process.events)
    print 'Total #events', num_events

    # start_date = datetime.datetime(2015, 9, 15)
    # fig = process.plot(start_date=start_date, user_limit=5,
    #                    num_samples=5000, time_unit='days',
    #                    label_every=1, seed=5)
    # fig.savefig("Figs/U_" + str(num_users) + "_E_" + str(num_events) + "_generated_intensity_trace.pdf")
    # plt.close(fig)

    return process

def infer(generated_process, alpha_0, mu_0, omega, num_users, vocab_types, num_particles):

    particle, norms = hdhp.infer(generated_process.events, alpha_0=alpha_0, mu_0=mu_0,
                                 omega=omega, num_particles=num_particles, seed=512, vocab_types=vocab_types)

    inferred_process = particle.to_process()

    # start_date = datetime.datetime(2015, 9, 15)

    # fig = inferred_process.plot(task_detail=True, num_samples=1000, seed=170,
    #                       time_unit='days', user_limit=5,
    #                       T_min=0, start_date=start_date, paper=True)
    #
    # fig.savefig("Figs/U_" + str(num_users) + "_E_" + str(len(generated_process.events)) + "_inferred_intensity_trace.pdf")
    # plt.close(fig)

    return inferred_process


def main():

    vocab_types = ['auths', 'docs']
    vocab_size = {'auths': 100, 'docs': 100}

    doc_min_length = {'auths': 10, 'docs': 10}
    doc_length = {'auths': 20, 'docs': 20}
    words_per_pattern = {'auths': 30, 'docs': 30}

    alpha_0 = (10, 0.2)
    mu_0 = (8, 0.25)
    omega = 5

    num_patterns = 25
    num_users = 30
    num_samples = 8000
    num_particles = 20

    start = timeit.default_timer()
    generated_process = generate(num_users, num_patterns, alpha_0, mu_0, omega, vocab_size, doc_min_length, doc_length, words_per_pattern, num_samples, vocab_types)
    print("Generation Time: " + str(timeit.default_timer() - start))

    start = timeit.default_timer()
    inferred_process = infer(generated_process, alpha_0, mu_0, omega, num_users, vocab_types, num_particles)
    print("Inference Time: " + str(timeit.default_timer() - start))

    num_events = len(generated_process.events)

    with open("Results/CM_U_" + str(num_users) + "_E_" + str(num_events) + "_P_" + str(num_patterns) + "_base_rates.tsv", "w") as fout:
        for key in generated_process.mu_per_user:
            fout.write("\t".join([str(key), str(generated_process.mu_per_user[key]), str(inferred_process.mu_per_user[key])]) + "\n")

    with open("Results/CM_U_" + str(num_users) + "_E_" + str(num_events) + "_P_" + str(num_patterns) + "_set_time_kernels.tsv" ,"w") as fout:
        for key in generated_process.time_kernels:
            fout.write("\t".join([str(key), str(generated_process.time_kernels[key])]) + "\n")
    #
    # with open("Results/CM_U_" + str(num_users) + "_E_" + str(num_events) + "_est_time_kernels.tsv", "w") as fout:
    #     for key in inferred_process.time_kernels:
    #         fout.write("\t".join([str(key), str(inferred_process.time_kernels[key])]) + "\n")


    # plot the base rates and the estimated alpha values
    plotMuScatterPlot(generated_process.mu_per_user, inferred_process.mu_per_user,
                      "Figs/CM_U_" + str(num_users) + "_E_" + str(
                          num_events) + "_P_" + str(num_patterns) + "_base_rates.pdf")

    trueLabs = [e[1] for e in generated_process.annotatedEventsIter()]
    predLabs = [e[1] for e in inferred_process.annotatedEventsIter()]

    print("True Labels Size: " + str(len(trueLabs)) + " seconds")
    print("predicted Labels Size: " + str(len(predLabs))+ " seconds")

    kernel_mappings = find_kernel_mapping(trueLabs, predLabs)

    generated_time_kernels = generated_process.time_kernels
    inferred_time_kernels = inferred_process.time_kernels

    new_inferred_time_kernels = {}

    for key in kernel_mappings:
        new_inferred_time_kernels[key] = inferred_time_kernels[kernel_mappings[key]]


    with open("Results/CM_U_" + str(num_users) + "_E_" + str(num_events) + "_est_time_kernels.tsv", "w") as fout:
        for key in new_inferred_time_kernels:
            fout.write("\t".join([str(key), str(new_inferred_time_kernels[key])]) + "\n")

    plotAlphaScatterPlot(generated_time_kernels, new_inferred_time_kernels,
                         "Figs/CM_U_" + str(num_users) + "_E_" + str(
                             num_events) + "_P_" + str(num_patterns) + "_time_kernels.pdf")

    with open("Results/CM_U_" + str(num_users) + "_E_" + str(num_events) + "_patterns.tsv", "w") as fout:
        for i in xrange(len(trueLabs)):
            fout.write("\t".join([str(trueLabs[i]), str(predLabs[i])]) + "\n")

    print ("NMI = " + str(normalized_mutual_info_score(trueLabs, predLabs)))


    # max_NMI, order = NMI.calculate_all_NMIs(trueLabs, predLabs, num_patterns)
    # print ("Max NMI: " + str(max_NMI))


if __name__ == "__main__":
    main ()


