import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcdefaults()
plt.rc('font', family='Georgia')

import json
import timeit
import numpy as np
import operator
from dateutil import relativedelta
import real_data_inference
from scipy import stats
from numpy import log as ln
from scipy.special import gammaln
import datetime
from scipy.special import gamma
from collections import Counter
import dill
import pickle
# import bigfloat
from operator import mul
import math


def plot_pattern_sizes(patterns_file_path, case, num_events, lead_author):
    from brokenaxes import brokenaxes

    with open(patterns_file_path) as patterns_file:
        lines = patterns_file.readlines()

    patterns = {}

    for line in lines:
        pattern = line.strip()

        if pattern not in patterns:
            patterns[pattern] = 0
        patterns[pattern] += 1

    sorted_patterns = sorted(patterns.items(), key=operator.itemgetter(1), reverse=True)
    y = []

    line = ""
    index = 0
    for item in sorted_patterns:
        y.append(item[1])
        index += 1
        line += str(index) + "," + str(item[1]) + "\n"

    x = range(len(y))

    with open("real_data_results/" + case + "/patterns_frequency_" + str(num_events) + "_" + lead_author + ".csv",
              "w") as output_file:
        output_file.write(line)

    # fig = plt.figure()
    # bax = brokenaxes(xlims=((0, 2000), (14000, 17000)), hspace=.05)
    # bax.hist(y)
    # bax.set_xlabel('Patterns')
    # bax.set_ylabel("Patterns' Size")
    # fig.savefig(output_file, bbox_inches='tight')
    # fig.clf()
    # plt.close(fig)
    output_file = "real_data_results/" + case + "/patterns_frequency_" + str(num_events) + "_" + lead_author + ".png"
    topic = "Size of all clusters (Setup " + case[-1] + ")"
    yticks = [100, 300, 500, 1000, 2000, 3000, 4000, 5000]
    ylim = [0, 6000]
    bar_plot(x, y, topic, "Clusters (Topics)", "Size", output_file, ylim=[0, 1900],
             yticks=[30, 100, 300, 500, 1000, 1900])


def memoize(f):
    class memodict(dict):
        __slots__ = ()

        def __missing__(self, key):
            self[key] = ret = f(key)
            return ret

    return memodict().__getitem__


@memoize
def _gammaln(x):
    return gammaln(x)


@memoize
def _ln(x):
    return ln(x)


def document_log_likelihood(dn_word_counts, count_dn, z_n, vocab_type, inferred_process):
    """Returns the log likelihood of document d_n to belong to cluster z_n.

    Note: Assumes a Gamma prior on the word distribution.
    """
    theta = inferred_process.theta_0[vocab_type][0]
    V = inferred_process.vocabulary_length[vocab_type]
    if z_n not in inferred_process.per_pattern_word_count_total[vocab_type]:
        count_zn_no_dn = 0
    else:
        count_zn_no_dn = inferred_process.per_pattern_word_count_total[vocab_type][z_n]
    # TODO: The code below works only for uniform theta_0. We should
    # put the theta that corresponds to `word`. Here we assume that
    # all the elements of theta_0 are equal
    gamma_numerator = _gammaln(count_zn_no_dn + V * theta)
    gamma_denominator = _gammaln(count_zn_no_dn + count_dn + V * theta)
    is_old_topic = z_n <= inferred_process._max_dish
    unique_words = len(dn_word_counts) == count_dn
    topic_words = None
    if is_old_topic:
        topic_words = inferred_process.per_pattern_word_counts[vocab_type][z_n]

    if unique_words:
        rest = [_ln(topic_words[word] + theta)
                if is_old_topic and word in topic_words
                else inferred_process._lntheta[vocab_type]
                for word in dn_word_counts]
    else:
        rest = [_gammaln(topic_words[word] + dn_word_counts[word] + theta) - _gammaln(topic_words[word] + theta)
                if is_old_topic and word in topic_words
                else _gammaln(dn_word_counts[word] + theta) - _gammaln(theta)
                for word in dn_word_counts]
    return gamma_numerator - gamma_denominator + sum(rest)


def calculate_perplexity(inferred_process_file_path, test_data_file_path, selected_authors, author_index, vocab_type):
    test_data_events = real_data_inference.json_file_to_events(test_data_file_path, vocab_type, 10, selected_authors,
                                                               author_index)

    test_data_events = test_data_events[0:1000]
    print("Test Dataset Size: " + str(len(test_data_events)))

    with open(inferred_process_file_path) as inferred_process_file:
        inferred_process = pickle.load(inferred_process_file)

    patterns_popularity = {}
    total_tables = 0

    for user in inferred_process.dish_on_table_per_user:
        user_tables = inferred_process.dish_on_table_per_user.get(user)
        total_tables += len(user_tables)

        for table in user_tables:
            dish = user_tables.get(table)
            if dish not in patterns_popularity:
                patterns_popularity[dish] = 0
            patterns_popularity[dish] += 1
    patterns_popularity = {dish: patterns_popularity[dish] * 1.0 / total_tables for dish in patterns_popularity}

    total_log_likelihood = 0
    test_data_size = 0
    patterns = patterns_popularity.keys()

    for event in test_data_events:
        t_n, d_n, u_n, q_n = event
        u_n = u_n[author_index]
        if u_n not in inferred_process.mu_per_user:
            continue
        test_data_size += 1
        d_n = {vocab_type: d_n[vocab_type].split() for vocab_type in d_n}
        dn_word_counts = {vocab_type: Counter(d_n[vocab_type]) for vocab_type in d_n}
        count_dn = {vocab_type: len(d_n[vocab_type]) for vocab_type in d_n}

        pattern_intensities = {}
        document_likelihood = {}
        event_prob = 0

        for pattern in patterns:
            alpha = inferred_process.time_kernels[pattern]

            popularity = patterns_popularity[pattern]
            intensity = popularity * inferred_process.mu_per_user[u_n]
            sum_intensity = 0
            if u_n in inferred_process.time_history_per_user:
                user_time_history = inferred_process.time_history_per_user[u_n]
                user_table_history = inferred_process.table_history_per_user[u_n]
                user_dish_on_table = inferred_process.dish_on_table_per_user[u_n]

                for index, time in enumerate(user_time_history):
                    dish = user_dish_on_table[user_table_history[index]]
                    if dish == pattern:
                        if t_n > time:
                            update_value = np.exp(-inferred_process.omega * (t_n - time))
                            sum_intensity += alpha * update_value
            intensity += sum_intensity
            pattern_intensities[pattern] = intensity

            temp_dll = {}
            for vocab_type in inferred_process.vocabulary_length:
                temp_dll[vocab_type] = document_log_likelihood(dn_word_counts[vocab_type], count_dn[vocab_type],
                                                               pattern, vocab_type, inferred_process)

            document_likelihood[pattern] = 1
            for vocab_type in temp_dll:
                document_likelihood[pattern] *= np.exp(temp_dll[vocab_type])

        sum_patterns_intensity = sum(pattern_intensities.values())
        for pattern in pattern_intensities:
            event_prob += ((pattern_intensities[pattern] / sum_patterns_intensity) * document_likelihood[pattern])

        total_log_likelihood += ln(event_prob)

    print("Total log likelihood: " + str(total_log_likelihood))

    perplexity = np.exp(-1.0 * total_log_likelihood / len(test_data_events))
    # print("Perplexity: " + str(perplexity))
    return perplexity


def scatter_plot(x, y, title, x_label, y_label, output_file_path, xlim=None, xticks=None):
    fig, ax = plt.subplots()
    ax.plot(x, y, 'bo')
    ax.set_xlabel(x_label, fontsize=12, fontweight="bold", fontname='Georgia')
    ax.set_ylabel(y_label, fontsize=12, fontweight="bold", fontname='Georgia')
    ax.set_title(title, fontweight="bold", fontname='Georgia', fontsize=16)
    ax.xaxis.set_tick_params(labelsize=14)
    ax.yaxis.set_tick_params(labelsize=14)
    if xticks is not None:
        ax.set_xticks(xticks)
    if xlim is not None:
        ax.set_xlim(xlim)
    fig.savefig(output_file_path, bbox_inches='tight')
    fig.clf()
    plt.close(fig)


def bar_plot(x, y, title, x_label, y_label, output_file_path, xlim=None, ylim=None, xticks=None, yticks=None, color="c",
             width=1 / 1.5):
    fig, ax = plt.subplots()

    plt.tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        right='off',
        labelright='off',
        labelbottom=False)  # lab

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.xaxis.set_tick_params(labelsize=14)
    ax.yaxis.set_tick_params(labelsize=14)
    # ax.set_xticks([50, 100, 300, 500, 1000, 2000, 3000])
    if xticks is not None:
        ax.set_xticks(xticks)
    if yticks is not None:
        ax.set_yticks(yticks)

    if xlim is not None:
        ax.set_xlim(xlim)

    if ylim is not None:
        ax.set_ylim(ylim)

    plt.xticks([])

    ax.bar(x, y, width, color=color, linewidth=0.4)
    ax.set_ylabel(y_label, fontsize=12, fontweight="bold", fontname='Georgia')
    ax.set_xlabel(x_label, fontsize=12, fontweight="bold", fontname='Georgia')
    ax.set_title(title, fontweight="bold", fontname='Georgia', fontsize=16)
    fig = plt.gcf()
    plt.close(fig)
    fig.savefig(output_file_path, bbox_inches='tight')
    fig.clf()
    plt.close(fig)


def coauthorship_per_topic(dataset_file_path, predictions_file_path, time_unit, selected_authors, case,
                           num_events, lead_author):
    """
    Finding coauthorship for each topic and saving it as a JSON file

    :param dataset_file_path: Dataset file path
    :param predictions_file_path: Predictions file path (indicates the pattern related to each sample)
    :param time_unit:
    :param selected_authors: Filtered authors (For example: authors with more than 1 papers)
    :param case: Indicates teh case we're running teh experiments (case1,2,3)
    :param num_events:
    :param lead_author: First or Last author
    :return: Nothing.
    """

    data = json.load(open(dataset_file_path))
    predictions = {}
    coauthorships = {}
    authors_topics = {}
    authors_names_mapping = {}

    with open(predictions_file_path) as predictions_file:
        lines = predictions_file.readlines()
        # print("Lines: " + str(len(lines)))
        for index, line in enumerate(lines):
            predictions[index] = line.strip()

    times = {}
    for identifier in data:
        times[identifier] = data.get(identifier).get(time_unit)

    sorted_times = sorted(times.items(), key=operator.itemgetter(1))

    index = 0
    for item in sorted_times:
        identifier = item[0]
        paper = data.get(identifier)
        authors_ids = paper.get('authors_ids')

        authors = paper['author']

        for i, author in enumerate(authors):
            authors_names_mapping[authors_ids[i]] = author

        if authors_ids[0] not in selected_authors and authors_ids[-1] not in selected_authors:
            continue

        topic = predictions[index]

        index += 1

        if topic not in coauthorships:
            coauthorships[topic] = {}

        for i in range(len(authors_ids)):
            a1 = authors_ids[i]
            # if a1 not in first_authors and a1 not in last_authors:
            #     continue

            if a1 not in authors_topics:
                authors_topics[a1] = {'name': authors[i], 'topics': []}
            if topic not in authors_topics[a1]['topics']:
                authors_topics.get(a1)['topics'].append(topic)

            if a1 not in coauthorships.get(topic):
                coauthorships[topic][a1] = {}

            for j in range(i + 1, len(authors_ids)):
                a2 = authors_ids[j]
                if a2 not in coauthorships[topic][a1]:
                    coauthorships[topic][a1][a2] = 0
                coauthorships[topic][a1][a2] += 1

                if a2 not in coauthorships.get(topic):
                    coauthorships[topic][a2] = {}
                if a1 not in coauthorships.get(topic)[a2]:
                    coauthorships[topic][a2][a1] = 0
                coauthorships[topic][a2][a1] += 1

    # Save the coauthorship network for all topics
    with open('real_data_results/' + case + '/coauthorships_per_topic_' + num_events + "_" + lead_author + '.json',
              'w') as output_file:
        print("Number of topics: " + str(len(coauthorships)))
        json.dump(coauthorships, output_file, indent=1)

    # Save mapping between authors names and their ids
    with open('real_data_results/Statistics/authors_ids_to_names.json', 'w') as output_file:
        print("Number of Authors: " + str(len(authors_names_mapping)))
        json.dump(authors_names_mapping, output_file, indent=1)

    # Save the topics for each author
    with open('real_data_results/' + case + '/authors_topics_' + num_events + "_" + lead_author + '.json',
              'w') as output_file:
        print("Number of Authors: " + str(len(authors_topics)) + '\n')
        json.dump(authors_topics, output_file, indent=1)

    print("_____________________________________________________________________________________")


def get_coauthorship_matrix(coauthorship_file_path, ids_to_names_file_path, topics, case, num_events, lead_author):
    """
    Create a CSV file of the coauthers per topic (use teh csv file to visualize the network).
    :param coauthorship_file_path: coauthorship file path (JSON file produced by running 'coauthorship_per_topic' function)
    :param ids_to_names_file_path: Maping author names to ids
    :param topics: Topics we want the network for
    :param case: The case we're running experiments for
    :param num_events: Number of events
    :param lead_author: First or last author
    :return: Nothing
    """
    topics_coauthorship = json.load(open(coauthorship_file_path))
    ids_to_names = json.load(open(ids_to_names_file_path))

    for topic in topics:
        print("Topic: " + topic)

        coauthorship = topics_coauthorship[topic]
        authors_ids = coauthorship.keys()
        print("Number of Authors: " + str(len(authors_ids)))
        line = ';'
        for a_id in authors_ids:
            author_name = ids_to_names[a_id]
            if ',' in author_name:
                split_name = author_name.split(',')
                author_name = split_name[1] + ' ' + split_name[0]
            line += author_name + ';'
        line = line[:-1]
        line += '\n'
        for author_id in authors_ids:
            author = ids_to_names[author_id]
            if ',' in author:
                split_name = author.split(',')
                author = split_name[1] + ' ' + split_name[0]
            line += author
            author_coauthorship = coauthorship[author_id]
            for a_id in authors_ids:
                if a_id not in author_coauthorship:
                    line += ';0'
                else:
                    line += ';' + str(author_coauthorship[a_id])
            line += '\n'

        with open(
                'real_data_results/' + case + '/coauthorship_matrix_' + topic + '_' + num_events + '_' + lead_author + '.csv',
                'w') as output_file:
            output_file.writelines(line.encode('utf-8'))
        print("********************************************************************************************")


def plot_papers_per_year(dataset_file_path):
    """
    This function plots number of papers published in each year
    :param dataset_file_path: Dataset File Path
    :return: Nothing.
    """
    start = timeit.default_timer()
    json_data = json.load(open(dataset_file_path))
    years = {}

    for identifier in json_data:
        paper = json_data[identifier]
        paper_year = paper['year']
        if paper_year in years:
            years[paper_year] += 1
        else:
            years[paper_year] = 1
    x = []
    y = []

    for key in years:
        x.append(int(key))
        y.append(years.get(key))

    output_file = "real_data_results/Statistics/papers_per_year.png"
    bar_plot(x, y, "Number of published papers per year", "Year", "Number of Papers", output_file)
    print("Plotting number of papers for each year in " + str(timeit.default_timer() - start) + " seconds")


def calculate_fitness_of_good(dataset_file_path, inferred_process_file_path, author_index, omega, lead_author):
    """
    Run Anderson Darling and KS tests
    :param dataset_file_path: Dataset file path
    :param inferred_process_file_path: Inferred process file path (pickle file)
    :param author_index: 0 for first authors and -1 for last authors
    :param omega: Kernel decay
    :param lead_author: First or last author
    :return: Nothing
    """
    with open(inferred_process_file_path) as inferred_process_file:
        inferred_process = pickle.load(inferred_process_file)

    vocab_types = ["title", "auths"]
    base_year = '2010'
    number_of_papers = 1
    # author_index = -1  # 0 for first authors and -1 for last authors

    first_authors, last_authors, all_authors = real_data_inference.get_authors_with_n_papers(dataset_file_path,
                                                                                             base_year,
                                                                                             number_of_papers)
    raw_events = real_data_inference.json_file_to_events(dataset_file_path, vocab_types, 10, all_authors, author_index)

    cases = {1: ([0], False),
             2: ([0, 1], False),
             3: ([0, 1], True)}
    user_integrals = {}


    for case in [3, 2, 1]:

        print "Case: {0}".format(case)
        indices, use_cousers = cases[case]

        types = ["docs", "auths"]

        # Inference
        types = [types[i] for i in indices]

        events = list()
        events_per_user = {}
        # Create the events from the dataset file

        if use_cousers:
            for index, event in enumerate(raw_events):
                events.append((event[0], {t: event[1][t] for t in types}, event[2], event[3]))
                for user in event[2]:
                    # user = str(user)
                    if user not in events_per_user:
                        events_per_user[user] = []
                    events_per_user[user].append(index)
        else:
            for index, event in enumerate(raw_events):
                events.append((event[0], {t: event[1][t] for t in types}, [event[2][author_index]], event[3]))
                user = event[2][author_index]
                # user = str(user)
                if user not in events_per_user:
                    events_per_user[user] = []
                events_per_user[user].append(index)

        mu_per_user = inferred_process.mu_per_user
        estimated_kernels = inferred_process.time_kernels
        predicted_labels = {}

        with open("real_data_results/" + "Case{0}".format(case) + "/months/title_patterns_" + str(
                len(events)) + '_' + lead_author + ".tsv") as patterns_file:
            lines = patterns_file.readlines()
            for index, line in enumerate(lines):
                predicted_labels[index] = int(line.strip())

        user_integrals[case] = calculate_transformed_points(events, events_per_user, mu_per_user, estimated_kernels,
                                                            predicted_labels, omega)
    plot_ks_test_results(user_integrals, 0.05, lead_author)
    plot_anderson_darling_test(user_integrals, 0.05, lead_author)


def plot_anderson_darling_test(user_integrals, threshold, lead_author):
    """
    Computes the Anderson Darling test and plot the results for the all three cases and saves the plot
    :param user_integrals: list of integrals of the intensity between two consecutive events per user (dict)
    :param threshold:
    :param lead_author: First or Last author (just used in the plot's name)
    :return:
    """

    reject_percentage = {case: 0 for case in user_integrals}

    for case in user_integrals:
        all_integrals = user_integrals[case]
        for user in all_integrals:
            if len(all_integrals.get(user)) < 2:
                continue
            statistic, critical_values, significance_level = stats.anderson(all_integrals.get(user), dist='expon')
            if statistic > critical_values[2]:
                reject_percentage[case] += 1

        reject_percentage[case] /= (len(all_integrals) * 1.0)

    fig, ax = plt.subplots()
    plt.show(block=False)
    ind = np.arange(1, 4)
    y = [reject_percentage[1], reject_percentage[2], reject_percentage[3]]
    print(y)

    pm, pc, pn = plt.bar(ind, y)
    pm.set_facecolor('r')
    pc.set_facecolor('g')
    pn.set_facecolor('b')
    # ax.set_xticks(ind)
    ax.set_xticklabels('')

    # Customize minor tick labels
    ax.set_xticks([1.4, 2.4, 3.4], minor=True)
    ax.set_xticklabels(['HDHP', 'HDHP-multi-vocab', 'multi_user_vocab'], minor=True)

    ax.set_ylim([0, 0.2])
    ax.set_ylabel('% Users Rejected')
    ax.set_title('Cases')
    ax.set_title('Anderson Test - ' + str(lead_author))
    fig = plt.gcf()
    plt.close(fig)
    fig.savefig("anderson_darling_TEST_" + lead_author + ".png")
    fig.clf()
    plt.close(fig)


def plot_ks_test_results(user_integrals, threshold, lead_author):
    """
    Computes the KS test and plot the results for the all three cases and saves the plot
    :param user_integrals: list of integrals of the intensity between two consecutive events per user (dict)
    :param threshold: threshold which is used in the test
    :param lead_author: First or Last author (just used in the plot's name)
    :return:
    """
    reject_percentage = {case: 0 for case in user_integrals}

    for case in user_integrals:
        all_integrals = user_integrals[case]
        for user in all_integrals:
            if len(all_integrals.get(user)) < 2:
                continue
            test_value, p_value = stats.kstest(all_integrals.get(user), 'expon')
            if p_value > threshold:
                reject_percentage[case] += 1

        reject_percentage[case] /= (len(all_integrals) * 1.0)
        reject_percentage[case] *= 100

    fig, ax = plt.subplots()
    plt.show(block=False)
    ind = np.arange(1, 4)
    y = [reject_percentage[1], reject_percentage[2], reject_percentage[3]]

    pm, pc, pn = plt.bar(ind, y)
    pm.set_facecolor('r')
    pc.set_facecolor('g')
    pn.set_facecolor('b')
    # ax.set_xticks(ind)
    ax.set_xticklabels('')

    # Customize minor tick labels
    ax.set_xticks([1.4, 2.4, 3.4], minor=True)
    ax.set_xticklabels(['HDHP', 'HDHP-multi-vocab', 'multi_user_vocab'], minor=True)

    ax.set_ylim([0, 30])
    ax.set_ylabel('% Users Rejected')
    ax.set_title('Cases')
    ax.set_title('KS Test - ' + str(lead_author))
    fig = plt.gcf()
    plt.close(fig)
    fig.savefig("KS_TEST_" + lead_author + ".png")
    fig.clf()
    plt.close(fig)


def calculate_transformed_points(events, events_per_user, mu_per_user, kernel_times, patterns, omega):
    """
    Computes the integral of the intensity between two consecutive events

    :param events: All events
    :param events_per_user: Events indices per user
    :param mu_per_user: Users' base rates
    :param kernel_times: Kernel times
    :param patterns: Patterns of samples
    :param omega: Kernel decay
    :return: A dictionary contains the integrals per user
    """
    user_integrals = {user: [] for user in events_per_user}
    print("number of users: {}".format(len(mu_per_user)))
    for user in events_per_user:

        user_mu = mu_per_user[user]
        user_events_indices = events_per_user[user]

        for i in range(len(user_events_indices) - 1):
            event1 = events[user_events_indices[i]]
            event2 = events[user_events_indices[i + 1]]
            t1 = event1[0]
            t2 = event2[0]
            integral = user_mu * (t2 - t1)
            first_part = 0
            second_part = 0
            event_index = user_events_indices[i]
            dish = patterns.get(event_index)
            second_part += (np.exp(-omega * (t2 - t1)) * kernel_times[dish])

            for index in range(i):
                event = events[user_events_indices[index]]
                time = event[0]
                event_index = user_events_indices[index]
                dish = patterns.get(event_index)
                first_part += (np.exp(-omega * (t1 - time)) * kernel_times[dish]) / -omega
                second_part += (np.exp(-omega * (t2 - time)) * kernel_times[dish]) / -omega

            integral += (second_part - first_part)
            user_integrals[user].append(integral)
    return user_integrals


def plot_popularity_vs_burstiness(inferred_process_file_path, num_events, case, lead_author, doc_type):
    """
    Computes popularity, and burstiness of each topic and plots popularity, burstiness, and popularity vs burstiness
    :param inferred_process_file_path: Inferred process file path (pickle file)
    :param num_events: Number of events
    :param case: The case we're running the result for.
    :param lead_author: First or Last author
    :param doc_type: Document type we use (topic or abstract)
    :return: Nothing.
    """
    with open(inferred_process_file_path) as inferred_process_file:
        inferred_process = pickle.load(inferred_process_file)

    dish_on_table_per_user = inferred_process.dish_on_table_per_user
    table_history_per_user = inferred_process.table_history_per_user

    popularity = {}
    burstiness_per_user = {}
    burstiness = {}
    total_tables = 0

    for user in dish_on_table_per_user:
        table_history = table_history_per_user.get(user)
        user_tables = dish_on_table_per_user.get(user)
        total_tables += len(user_tables)

        for table in user_tables:
            dish = user_tables.get(table)
            if dish not in popularity:
                popularity[dish] = 0
                burstiness_per_user[dish] = {}
            popularity[dish] += 1

            if user not in burstiness_per_user.get(dish):
                burstiness_per_user[dish][user] = 0
            burstiness_per_user[dish][user] += (table_history.count(int(table)) - 1)

    sorted_pattern_popularity = sorted(popularity.iteritems(),
                                       key=lambda x: (x[1], x[0]),
                                       reverse=True)

    n = 50
    top_popular_patterns = sorted_pattern_popularity[:n]

    y = []

    for item in top_popular_patterns:
        y.append(item[1] * 1.0 / total_tables)

    x = range(len(y))
    output_file_path = "real_data_results/" + case + "/" + doc_type + "_popularity_" + str(
        num_events) + "_" + lead_author + ".png"

    # Plot patterns' popularity
    bar_plot(x, y, "Pattern's Popularity (Setup " + case[-1] + ")", "Patterns", "Popularity", output_file_path)

    patterns = {}
    predicted_labels = [e[1] for e in inferred_process.annotatedEventsIter()]
    for index, topic in enumerate(predicted_labels):

        if topic not in patterns:
            patterns[topic] = 1
        else:
            patterns[topic] += 1

    for dish in burstiness_per_user:
        if patterns[dish] > 15:

            avg_burstiness = sum(burstiness_per_user.get(dish).values()) * 1.0 / len(
                burstiness_per_user.get(dish))  # len(dish_on_table_per_user)
            burstiness[dish] = avg_burstiness

    y = []
    z = []

    for dish in burstiness:
        y.append(burstiness.get(dish))
        z.append(popularity.get(dish) * 1.0 / total_tables)

    x = range(len(y))
    # Plot patterns' burstiness
    output_file_path = "real_data_results/" + case + "/" + doc_type + "_burstiness_" + str(
        num_events) + "_" + lead_author + ".png"
    bar_plot(x, y, "Topics' Burstiness (Setup " + case[-1] + ")", 'Patterns', 'Burstiness', output_file_path)

    # Plot popularity vs burstiness
    output_file_path = "real_data_results/" + case + "/" + doc_type + "_burstiness_vs_popularity_" + str(
        num_events) + "_" + lead_author + ".png"
    # xticks = [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    scatter_plot(y, z, "Busrtiness vs Popularity (Setup " + case[-1] + ")", 'Burstiness', 'Popularity',
                 output_file_path, xlim=[-0.001, max(y) + 0.001])

    # sorted_burstiness = sorted(burstiness.iteritems(),
    #                            key=lambda x: (x[1], x[0]), reverse=True)

    # estimated_kernels = inferred_process.time_kernels
    # y = []
    # z = []
    # for dish in popularity:
    #     y.append(estimated_kernels[dish])
    #     z.append(popularity[dish] * 1.0 / total_tables)
    #
    # output_file_path = "real_data_results/" + case + "/" + doc_type + "_time_kernels_vs_popularity_" + str(
    #     num_events) + "_" + lead_author + ".png"
    # # xticks = [-1, 0, 2, 4, 6, 8]
    # scatter_plot(y, z, "Kernels vs Popularity (Setup " + case[-1] + ")", 'Burstiness', 'Popularity',
    #              output_file_path)


def plot_papers_per_user(dataset_file_path, base_year):
    """
    Plot number of papers per author
    :param dataset_file_path: Dataset file path
    :param base_year: We want results for papers are published after this base year
    :return:
    """
    from datetime import datetime

    data = json.load(open(dataset_file_path))
    base_time = datetime.strptime(base_year + '-01-01', '%Y-%m-%d')
    times = {}
    first_author_papers = {}
    first_author_times = {}
    last_author_papers = {}
    last_author_times = {}
    author_papers = {}
    author_times = {}

    for identifier in data:

        paper = data.get(identifier)
        authors_ids = paper['authors_ids']

        for author_id in authors_ids:
            if author_id not in author_papers:
                author_papers[author_id] = 0
            author_papers[author_id] += 1

        if authors_ids[0] not in first_author_papers:
            first_author_papers[authors_ids[0]] = 0
        first_author_papers[authors_ids[0]] += 1
        if authors_ids[-1] not in last_author_papers:
            last_author_papers[authors_ids[-1]] = 0
        last_author_papers[authors_ids[-1]] += 1

        if int(paper['year']) > int(base_year):
            paper_time = datetime.strptime(paper['date'][0], '%Y-%m-%d')
            diff = relativedelta.relativedelta(paper_time, base_time)
            num_months = diff.years * 12 + diff.months
            times[identifier] = num_months

    sorted_times = sorted(times.items(), key=operator.itemgetter(1))
    for item in sorted_times:
        paper = data.get(item[0])
        authors_ids = paper['authors_ids']

        for author_id in authors_ids:
            if author_id not in author_times:
                author_times[author_id] = []
            author_times[author_id].append(item[1])

        if authors_ids[0] not in first_author_times:
            first_author_times[authors_ids[0]] = []
        first_author_times[authors_ids[0]].append(item[1])

        if authors_ids[-1] not in last_author_times:
            last_author_times[authors_ids[-1]] = []
        last_author_times[authors_ids[-1]].append(item[1])

    y = []
    for user in first_author_times:
        user_times = first_author_times.get(user)
        if len(user_times) < 2:
            continue
        if max(user_times) - min(user_times) == 0:
            continue
        temp = len(user_times) * 1.0 / (max(user_times) - min(user_times))
        y.append(temp * 6)
    y.sort(reverse=True)
    x = range(len(y))

    title = 'Avg num of papers in 6 months per unique first author ( since ' + base_year + ')'
    output_file = "real_data_results/Statistics/first_authors_papers_6_months.png"
    bar_plot(x, y, title, 'Authors', 'Number of papers', output_file, color="#408000")

    sorted_numbers = sorted(first_author_papers.iteritems(), key=operator.itemgetter(1), reverse=True)
    y = []
    for item in sorted_numbers:
        y.append(item[1])

    x = range(len(y))
    title = 'Number of published papers per unique first author ( since ' + base_year + ')'
    output_file = "real_data_results/Statistics/first_authors_papers.png"
    bar_plot(x, y, title, 'Authors', 'Number of papers', output_file, color="#408000")

    sorted_numbers = sorted(last_author_papers.iteritems(), key=operator.itemgetter(1), reverse=True)
    y = []
    for item in sorted_numbers:
        y.append(item[1])

    x = range(len(y))

    title = 'Number of published papers per unique last author ( since ' + base_year + ')'
    output_file = "real_data_results/Statistics/last_authors_papers.png"
    bar_plot(x, y, title, 'Authors', 'Number of papers', output_file, color="#408000")

    sorted_numbers = sorted(author_papers.iteritems(), key=operator.itemgetter(1), reverse=True)
    y = []
    for item in sorted_numbers:
        y.append(item[1])

    x = range(len(y))
    title = 'Number of published papers per author ( since ' + base_year + ')'
    output_file = "real_data_results/Statistics/authors_papers.png"
    bar_plot(x, y, title, 'Authors', 'Number of papers', output_file, color="#408000")


def plot_intensity(inferred_process_file_path, users, num_events, case, lead_author, patterns=None):

    """
    Plots user intensities
    :param inferred_process_file_path: Inferred process file path (pickle file)
    :param users: A list of users we want to plot their intensities for some patterns
    :param num_events: Number of events
    :param case: The case we're running results for
    :param lead_author: First or Last author
    :param patterns: The patterns we want to plot the intensities per user
    :return: Nothing.
    """
    if len(users) == 0:
        users = None
    with open(inferred_process_file_path) as inferred_process_file:
        inferred_process = pickle.load(inferred_process_file)

    patterns_popularity = {}
    total_tables = 0

    # Calculates patterns' popularity
    for user in inferred_process.dish_on_table_per_user:
        user_tables = inferred_process.dish_on_table_per_user.get(user)
        total_tables += len(user_tables)

        for table in user_tables:
            dish = user_tables.get(table)
            if dish not in patterns_popularity:
                patterns_popularity[dish] = 0
            patterns_popularity[dish] += 1
    patterns_popularity = {dish: patterns_popularity[dish] * 1.0 / total_tables for dish in patterns_popularity}
    inferred_process.pattern_popularity = patterns_popularity

    for user in inferred_process.time_history_per_user:
        temp = inferred_process.time_history_per_user[user]
        temp = [x * 6.0 for x in temp]
        inferred_process.time_history_per_user[int(user)] = temp

    start_date = datetime.datetime(2010, 1, 1)
    fig = inferred_process.plot(start_date=start_date, users=users, user_limit=5,
                                num_samples=1000, time_unit='months',
                                label_every=1, seed=5)
    fig.savefig("real_data_results/" + case + "/U_" + str(
        num_events) + "_generated_intensity_trace" + "_" + lead_author + ".pdf")
    plt.close(fig)


def average_published_time_per_user(dataset_file_path, base_year):
    from datetime import datetime

    data = json.load(open(dataset_file_path))

    base_time = datetime.strptime(base_year + '-01-01', '%Y-%m-%d')
    times = {}

    for identifier in data:

        paper = data.get(identifier)

        if int(paper['year']) > int(base_year):
            paper_time = datetime.strptime(paper['date'][0], '%Y-%m-%d')
            diff = relativedelta.relativedelta(paper_time, base_time)
            num_months = diff.years * 12 + diff.months
            times[identifier] = num_months

    sorted_times = sorted(times.items(), key=operator.itemgetter(1))

    first_author_times = {}
    last_author_times = {}

    for item in sorted_times:
        identifier = item[0]
        paper = data.get(identifier)

        authors_ids = paper.get('authors_ids')

        first_author = authors_ids[0]
        last_author = authors_ids[-1]

        if first_author not in first_author_times:
            first_author_times[first_author] = []
        first_author_times.get(first_author).append(item[1])

        if last_author not in last_author_times:
            last_author_times[last_author] = []
        last_author_times.get(last_author).append(item[1])

    print("Unique first authors: " + str(len(first_author_times)))
    print("Unique last authors: " + str(len(last_author_times)))

    first_author_papers = {}
    last_author_papers = {}

    first_author_avg_numbers = {}
    last_author_avg_numbers = {}

    for author in first_author_times:
        times = first_author_times.get(author)
        if len(times) < 2:
            continue
        avg_months = 0

        for i in range(len(times) - 1):
            avg_months += (times[i + 1] - times[i])

        avg_months /= len(times)
        first_author_papers[author] = avg_months
        if avg_months not in first_author_avg_numbers:
            first_author_avg_numbers[avg_months] = 1
        else:
            first_author_avg_numbers[avg_months] += 1

    print("Average num of months between two papers for all first authors: " + str(
        sum(first_author_papers.values()) / len(first_author_papers)) + "\n")
    x = first_author_avg_numbers.keys()
    y = [first_author_avg_numbers.get(key) for key in x]

    width = 1 / 1.5
    fig, ax = plt.subplots()

    ax.bar(x, y, width, color="c", linewidth=0.4)
    ax.set_ylabel('Number of authors')
    ax.set_xlabel('Avg num of months')
    ax.set_title('Avg num of months between two papers (for first authors)')
    fig = plt.gcf()
    plt.close(fig)
    fig.savefig("real_data_results/Statistics/first_authors_time_interval.png")
    fig.clf()
    plt.close(fig)

    for author in last_author_times:
        times = last_author_times.get(author)
        if len(times) < 2:
            continue
        avg_months = 0

        for i in range(len(times) - 1):
            avg_months += (times[i + 1] - times[i])

        avg_months /= len(times)
        last_author_papers[author] = avg_months
        if avg_months not in last_author_avg_numbers:
            last_author_avg_numbers[avg_months] = 1
        else:
            last_author_avg_numbers[avg_months] += 1

    print("Average num of months between two papers for all last authors: " + str(
        sum(last_author_papers.values()) / len(last_author_papers)) + "\n")

    x = last_author_avg_numbers.keys()
    y = [last_author_avg_numbers.get(key) for key in x]

    output_file = "real_data_results/Statistics/last_authors_time_interval.png"
    title = 'Avg num of months between two papers (for last authors)'
    bar_plot(x, y, title, 'Avg num of months', 'Number of authors', output_file, color="#408000")


###########

def calculate_fitness_of_good_test(dataset_file_path, num_events, author_index, omega, lead_author):
    """
    Run Anderson Darling and KS tests
    :param dataset_file_path: Dataset file path
    :param inferred_process_file_path: Inferred process file path (pickle file)
    :param author_index: 0 for first authors and -1 for last authors
    :param omega: Kernel decay
    :param lead_author: First or last author
    :return: Nothing
    """

    vocab_types = ["title", "auths"]
    base_year = '2010'
    number_of_papers = 1
    # author_index = -1  # 0 for first authors and -1 for last authors

    first_authors, last_authors, all_authors = real_data_inference.get_authors_with_n_papers(dataset_file_path,
                                                                                             base_year,
                                                                                             number_of_papers)
    raw_events = real_data_inference.json_file_to_events(dataset_file_path, vocab_types, 10, all_authors, author_index)

    cases = {1: ([0], False),
             2: ([0, 1], False),
             3: ([0, 1], True)}
    user_integrals = {}


    for case in [3, 2, 1]:

        inferred_process_file_path = "real_data_results/Case" + str(case) + "/Inference/title_inferred_process_" + num_events + "_" + lead_author + ".pkl"
        with open(inferred_process_file_path) as inferred_process_file:
            inferred_process = pickle.load(inferred_process_file)

        print "Case: {0}".format(case)
        indices, use_cousers = cases[case]

        types = ["docs", "auths"]

        # Inference
        types = [types[i] for i in indices]

        events = list()
        events_per_user = {}
        # Create the events from the dataset file

        if use_cousers:
            for index, event in enumerate(raw_events):
                events.append((event[0], {t: event[1][t] for t in types}, event[2], event[3]))
                for user in event[2]:
                    # user = str(user)
                    if user not in events_per_user:
                        events_per_user[user] = []
                    events_per_user[user].append(index)
        else:
            for index, event in enumerate(raw_events):
                events.append((event[0], {t: event[1][t] for t in types}, [event[2][author_index]], event[3]))
                user = event[2][author_index]
                # user = str(user)
                if user not in events_per_user:
                    events_per_user[user] = []
                events_per_user[user].append(index)

        mu_per_user = inferred_process.mu_per_user
        estimated_kernels = inferred_process.time_kernels
        predicted_labels = {}

        with open("real_data_results/" + "Case{0}".format(case) + "/months/title_patterns_" + str(
                len(events)) + '_' + lead_author + ".tsv") as patterns_file:
            lines = patterns_file.readlines()
            for index, line in enumerate(lines):
                predicted_labels[index] = int(line.strip())

        user_integrals[case] = calculate_transformed_points(events, events_per_user, mu_per_user, estimated_kernels,
                                                            predicted_labels, omega)
    plot_ks_test_results(user_integrals, 0.05, lead_author)
    plot_anderson_darling_test(user_integrals, 0.05, lead_author)

#################
def main():
    base_year = '2010'
    time_unit = 'time_half_year'
    number_of_papers = 1
    omega = 5

    dataset_file_path = "../Real_Dataset/data_after_year_2010.json"


    test_data_file_path = "../Real_Dataset/test_data_after_2016.json"
    train_data_file_path = "../Real_Dataset/train_data_before_2016.json"
    ids_to_names_file_path = "real_data_results/Statistics/authors_ids_to_names.json"
    vocab_types = ["title", "auths"]

    # first_authors, last_authors, all_authors = real_data_inference.get_authors_with_n_papers(dataset_file_path,
    #                                                                                          base_year,
    #                                                                                          number_of_papers)


    num_events = '35344'
    doc_type = "title"

    for case in ['Case3']:
        print("Setup: " + case)

        author_index = 0
        for lead_author in ["first_author"]:

            inferred_process_file_path = "real_data_results/" + case + "/Inference/title_inferred_process_" + num_events + "_" + lead_author + ".pkl"
            patterns_file_path = "real_data_results/" + case + "/months/title_patterns_" + num_events + "_" + lead_author + ".tsv"

            # calculate_fitness_of_good(dataset_file_path, inferred_process_file_path, author_index, omega, lead_author)
            calculate_fitness_of_good_test(dataset_file_path, num_events, author_index, omega, lead_author)
            # calculate_perplexity(inferred_process_file_path, test_data_file_path, all_authors, author_index, vocab_types)
            #
            # plot_popularity_vs_burstiness(inferred_process_file_path, num_events, case, lead_author, doc_type)

            coauthorship_file_path = "real_data_results/" + case + '/coauthorships_per_topic_' + num_events + "_"  + lead_author + '.json'
            # coauthorship_per_topic(dataset_file_path, patterns_file_path, time_unit, all_authors, case, num_events, lead_author)
            # get_coauthorship_matrix(coauthorship_file_path, ids_to_names_file_path, ['457'], case, num_events, lead_author)


if __name__ == "__main__":
    main()
