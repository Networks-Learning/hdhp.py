import matplotlib

matplotlib.use('Agg')
import seaborn as sns
import matplotlib.pyplot as plt

plt.rcdefaults()

import notebook_helpers

import argparse
import datetime
import hdhp

# sns.set(color_codes=True)
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

import pandas as pd


def plot_synthetic_stats(file_name):
    with open(file_name + "_base_rates.tsv") as base_rates_file, open(
                    file_name + "_est_time_kernels.tsv") as estimated_kernel_file, open(
                file_name + "_set_time_kernels.tsv") as set_time_kernels_file, open(
                file_name + "_patterns.tsv") as patterns_file:

        # Plot Base Rates

        lines = base_rates_file.readlines()
        x = []
        y = []

        for line in lines:
            temp = line.split('\t')
            x.append(float(temp[1]))
            y.append(float(temp[2]))
        min_x = min(x)
        max_x = max(x)

        z = np.linspace(int(min_x) + 0.4, max_x)

        fig, ax = plt.subplots(facecolor='white', figsize=(5, 5))
        ax.plot(z, z, color='#616A6B', alpha=0.5)

        ax.scatter(x, y, edgecolors='#6076E1', color='#A2AEE7', alpha=0.6)

        ax.spines['bottom'].set_color('black')
        ax.set_xlim([0, 5])
        ax.set_ylim([0, 5])
        # ax.scatter(x, y, edgecolors='#ADB5DB', color='#A2AEE7', alpha=0.6, s=25)
        ax.set_title('Base Rates')

        ax.set_ylabel('Inferred Value', fontsize=20)  # , fontweight="bold")
        ax.set_xlabel('True Value', fontsize=20)  # , fontweight="bold")
        # removing top and right borders
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # ax.spines['left'].set_linewidth(2)
        # ax.xaxis.set_tick_params(labelsize=14)
        # ax.yaxis.set_tick_params(labelsize=14)
        sns.set_style("ticks")
        sns.despine(offset=10, trim=True)
        sns.set_context("paper", font_scale=1.5, rc={"lines.linewidth": 2.5})

        # plt.show()

        temp = file_name.split('/')[-1][0:-4]
        plt.savefig("Results/Figs/" + temp + "base_rates.pdf", bbox_inches='tight')
        plt.clf()
        plt.close()

        # Plot time kernels
        true_labels = []
        pred_labels = []
        patterns_lines = patterns_file.readlines()

        for line in patterns_lines:
            split_line = line.split('\t')
            true_labels.append(split_line[0])
            pred_labels.append(split_line[1].strip())

        generated_time_kernels = {}
        lines = set_time_kernels_file.readlines()
        for line in lines:
            split_line = line.split('\t')
            generated_time_kernels[split_line[0]] = float(split_line[1])


        estimated_time_kernels = {}
        lines = estimated_kernel_file.readlines()
        for line in lines:
            split_line = line.split('\t')
            estimated_time_kernels[split_line[0]] = float(split_line[1])

        kernel_mappings = find_kernel_mapping(true_labels, pred_labels)
        new_inferred_time_kernels = {}

        for key in kernel_mappings:
            new_inferred_time_kernels[key] = estimated_time_kernels[kernel_mappings[key]]

        x = []
        y = []

        for key in kernel_mappings:
            x.append(generated_time_kernels[key])
            y.append(new_inferred_time_kernels[key])


        min_x = min(x)
        max_x = max(x)

        z = np.linspace(int(min_x) + 0.4, max_x + 0.2)

        fig, ax = plt.subplots(facecolor='white', figsize=(5, 5))
        ax.plot(z, z, color='#616A6B', alpha=0.5)

        ax.scatter(x, y, edgecolors='#6076E1', color='#A2AEE7', alpha=0.6)

        ax.spines['bottom'].set_color('black')
        ax.set_xlim([0, 4])
        ax.set_ylim([0, 4])
        ax.set_xticks([0,1,2,3,4])
        ax.set_yticks([0, 1, 2, 3, 4])

        ax.set_ylabel('Inferred Value', fontsize=20)  # , fontweight="bold")
        ax.set_xlabel('True Value', fontsize=20)  # , fontweight="bold")
        ax.set_title('Time Kernels')
        # removing top and right borders
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        sns.set_style("ticks")
        sns.despine(offset=10, trim=True)
        sns.set_context("paper", font_scale=1.5, rc={"lines.linewidth": 2.5})

        temp = file_name.split('/')[-1][0:-4]
        plt.savefig("Results/Figs/" + temp + "time_kernels.pdf", bbox_inches='tight')
        plt.clf()
        plt.close()


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


def plotMuScatterPlot(xdict, ydict, outFile):
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

    fig = ax.get_figure()
    fig.savefig(outFile)
    fig.clf()
    plt.close(fig)


def plotAlphaScatterPlot(xdict, ydict, outFile):
    xkeys = xdict.keys()
    ykeys = ydict.keys()

    inter = set(xkeys) & set(ykeys)

    x = pd.Series([xdict[k] for k in inter], name="True Value")
    y = pd.Series([ydict[k] for k in inter], name="Inferred Value")

    max_axis_value = max(max(x), max(y)) + 1
    min_axis_value = min(min(x), min(y)) - 1

    z = np.linspace(int(min_axis_value), 6)
    sns.plt.plot(z, z + 0, linestyle='solid')

    ax = sns.regplot(x=x, y=y, marker="o", fit_reg=False)
    ax.set(title="Fig Title: Kernel Parameter")

    sns.plt.ylim(min_axis_value, 6)
    sns.plt.xlim(min_axis_value, 6)

    # sns.plt.ylim(min_axis_value, max_axis_value)
    # sns.plt.xlim(min_axis_value, max_axis_value)

    fig = ax.get_figure()
    fig.savefig(outFile)
    fig.clf()
    plt.close(fig)


def generate(num_users, num_patterns, alpha_0, mu_0, omega, vocab_size, doc_min_length, doc_length, words_per_pattern,
             num_samples, vocab_types):
    start = timeit.default_timer()
    vocabulary = {}
    for vocab_type in vocab_types:
        vocabulary[vocab_type] = [vocab_type + str(i) for i in
                                  range(vocab_size[vocab_type])]  # the `words` of our documents

    process = hdhp.HDHProcess(num_patterns=num_patterns, alpha_0=alpha_0, num_users=num_users, vocab_types=vocab_types,
                              mu_0=mu_0, vocabulary=vocabulary, doc_length=doc_length, doc_min_length=doc_min_length,
                              omega=omega, words_per_pattern=words_per_pattern,
                              random_state=12, generate=True)

    # overlap = notebook_helpers.compute_pattern_overlap(process, vocab_types)
    # for vocab_type in vocab_types:
    #     ax = sns.distplot(overlap[vocab_type], kde=True, norm_hist=True, axlabel='Content overlap')
    #     fig = ax.get_figure()
    #     fig.savefig("Figs/Pattern_Overlaps" + str(vocab_type) + "_" + str(num_patterns) + "_pattern_overlaps.pdf")
    #     fig.clf()
    #     plt.close(fig)

    process.reset()  # removes any previously generated data
    process.sample_user_events(min_num_events=100,
                               max_num_events=num_samples,
                               t_max=365)
    for cluster in process.dish_counters:
        print("Cluster " + str(cluster) + " : " + str(process.dish_counters[cluster]))

    num_events = len(process.events)
    print 'Total #events', num_events
    print("Generation Time: " + str(timeit.default_timer() - start) + " seconds")

    # start_date = datetime.datetime(2015, 9, 15)
    # fig = process.plot(start_date=start_date, user_limit=5,
    #                    num_samples=5000, time_unit='days',
    #                    label_every=1, seed=5)
    # fig.savefig("Figs/U_" + str(num_users) + "_E_" + str(num_events) + "_generated_intensity_trace.pdf")
    # plt.close(fig)

    return process


def infer(generated_process, alpha_0, mu_0, omega, num_users, vocab_types, num_particles):
    start = timeit.default_timer()
    particle, norms = hdhp.infer(generated_process.events, alpha_0=alpha_0, mu_0=mu_0,
                                 omega=omega, num_particles=num_particles, seed=512, vocab_types=vocab_types)
    print("Inference Time: " + str(timeit.default_timer() - start) + " seconds")

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

    file_name = "../Synthetic_Results/CM_U_200_E_250560_P_50"
    # plot_synthetic_stats(file_name)
    #
    # return
    vocab_types = ['auths', 'docs']
    vocab_size = {'auths': 100, 'docs': 250}

    doc_min_length = {'auths': 10, 'docs': 20}
    doc_length = {'auths': 20, 'docs': 30}
    words_per_pattern = {'auths': 30, 'docs': 40}

    alpha_0 = (10, 0.2)
    mu_0 = (8, 0.25)
    omega = 5

    num_patterns = 20
    num_users = 20
    num_samples = 10000
    num_particles = 20

    print("****************************************")
    print(" Number of expected events: " + str(num_samples))
    print(" Number of users: " + str(num_users))
    print(" Number of patterns: " + str(num_patterns))
    print(" Number of particles: " + str(num_particles))
    print("****************************************")
    print

    generated_process = generate(num_users, num_patterns, alpha_0, mu_0, omega, vocab_size, doc_min_length, doc_length,
                                 words_per_pattern, num_samples, vocab_types)

    inferred_process = infer(generated_process, alpha_0, mu_0, omega, num_users, vocab_types, num_particles)

    num_events = len(generated_process.events)

    trueLabs = [e[1] for e in generated_process.annotatedEventsIter()]
    predLabs = [e[1] for e in inferred_process.annotatedInfEventsIter()]

    print("True Labels Size: " + str(len(trueLabs)))
    print("predicted Labels Size: " + str(len(predLabs)))

    with open("Results/CM_U_" + str(num_users) + "_E_" + str(num_events) + "_P_" + str(
            num_patterns) + "_base_rates.tsv", "w") as fout:
        for key in generated_process.mu_per_user:
            fout.write("\t".join(
                [str(key), str(generated_process.mu_per_user[key]), str(inferred_process.mu_per_user[key])]) + "\n")

    with open("Results/CM_U_" + str(num_users) + "_E_" + str(num_events) + "_P_" + str(
            num_patterns) + "_set_time_kernels.tsv", "w") as fout:
        for key in generated_process.time_kernels:
            fout.write("\t".join([str(key), str(generated_process.time_kernels[key])]) + "\n")

    with open("Results/CM_U_" + str(num_users) + "_E_" + str(num_events) + "_est_time_kernels.tsv", "w") as fout:
        for key in inferred_process.time_kernels:
            fout.write("\t".join([str(key), str(inferred_process.time_kernels[key])]) + "\n")

    # with open("Results/CM_U_" + str(num_users) + "_E_" + str(num_events) + "_time_kernels.tsv", "w") as fout:
    #     print (generated_process.time_kernels.keys())
    #     print(inferred_process.time_kernels.keys())
    #     for key in inferred_process.time_kernels:
    #         fout.write("\t".join(
    #             [str(key), str(generated_process.time_kernels[key]), str(inferred_process.time_kernels[key])]) + "\n")



    # plot the base rates and the estimated alpha values
    plotMuScatterPlot(generated_process.mu_per_user, inferred_process.mu_per_user,
                      "Results/Figs/CM_U_" + str(num_users) + "_E_" + str(
                          num_events) + "_P_" + str(num_patterns) + "_base_rates.pdf")


    kernel_mappings = find_kernel_mapping(trueLabs, predLabs)

    generated_time_kernels = generated_process.time_kernels
    inferred_time_kernels = inferred_process.time_kernels

    new_inferred_time_kernels = {}

    for key in kernel_mappings:
        new_inferred_time_kernels[key] = inferred_time_kernels[kernel_mappings[key]]

    plotAlphaScatterPlot(generated_time_kernels, new_inferred_time_kernels,
                         "Results/Figs/CM_U_" + str(num_users) + "_E_" + str(
                             num_events) + "_P_" + str(num_patterns) + "_time_kernels.pdf")

    with open("Results/CM_U_" + str(num_users) + "_E_" + str(num_events) + "_patterns.tsv", "w") as fout:
        for i in xrange(len(trueLabs)):
            fout.write("\t".join([str(trueLabs[i]), str(predLabs[i])]) + "\n")

    print ("NMI = " + str(normalized_mutual_info_score(trueLabs, predLabs)))


if __name__ == "__main__":
    main()
