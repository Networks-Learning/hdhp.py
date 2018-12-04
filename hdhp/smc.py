"""
    smc
    ~~~

    Provides an implementation of an inference algorithm for the Hierarchical
    Dirichlet-Hawkes process, based on a sequential Monte-Carlo with particles.

    :copyright: 2016 Charalampos Mavroforakis, <cmav@bu.edu> and contributors.
    :license: ISC
"""
from __future__ import division, print_function

import tempfile
from collections import Counter, defaultdict
from copy import copy
from time import time

from numpy import log as ln, array, exp
from numpy.random import RandomState
from scipy.misc import logsumexp
from scipy.special import gammaln
from utils import copy_dict, weighted_choice

from hdhp import HDHProcess

maxint = 10000000


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


class InferenceParameters:
    """This class collects all the parameters required for model inference.

    Its main use is to help keep the code clean.
    """

    def __init__(self, alpha_0, mu_0, omega, beta, theta_0,
                 threads, num_particles,
                 particle_weight_threshold, resample_every,
                 update_kernels, mu_rate,
                 keep_alpha_history, progress_file, seed,
                 vocabulary, users, author_index):
        self.alpha_0 = alpha_0
        self.mu_0 = mu_0
        self.omega = omega
        self.beta = beta
        self.theta_0 = theta_0
        self.threads = threads
        self.num_particles = num_particles
        self.particle_weight_threshold = particle_weight_threshold
        self.resample_every = resample_every
        self.update_kernels = update_kernels
        self.mu_rate = mu_rate
        self.keep_alpha_history = keep_alpha_history
        self.progress_file = progress_file
        self.seed = seed
        self.vocabulary = vocabulary
        self.users = users
        self.author_index = author_index


class Particle(object):
    def __init__(self, vocabulary_length, users, time_kernels=None,
                 alpha_0=(2, 2), mu_0=1, theta_0=None,
                 seed=None, logweight=0, update_kernels=False, uid=0,
                 omega=1, beta=1, keep_alpha_history=False, mu_rate=0.6, author_index=0):
        self.vocabulary_length = vocabulary_length
        self.vocabulary = None
        self.seed = seed
        self.prng = RandomState(self.seed)
        self.first_observed_time = {}
        self.first_observed_user_time = {}
        self.per_topic_word_counts = {vocab_type: {} for vocab_type in vocabulary_length}
        self.per_topic_word_count_total = {vocab_type: {} for vocab_type in vocabulary_length}
        self.time_kernels = {}
        self.alpha_0 = alpha_0
        self.mu_0 = mu_0
        self.theta_0 = {vocab_type: array(theta_0[vocab_type]) for vocab_type in theta_0}
        self._lntheta = {vocab_type: _ln(theta_0[vocab_type][0]) for vocab_type in theta_0}
        self.logweight = logweight
        self.update_kernels = update_kernels
        self.uid = uid
        self.num_events = 0
        self.topic_previous_event = None
        # The following are for speed optimization purposes
        # A struture to save the total intensity of a topic
        # up to the most recent event t_i of that topic.
        # It will be used to measure the total intensity at
        # any time after t_i
        self._Qn = None
        self.omega = omega
        self.beta = beta
        self.num_users = len(users)
        self.users = users
        self.max_users_id = max(users) + 1
        self.keep_alpha_history = keep_alpha_history

        # self.user_table_cache = {}
        self.dish_on_table_per_user = {}
        # self.dish_on_table_todelete = {}
        self.dish_counters = {}
        self._max_dish = -1
        # self.total_tables = 0
        self.total_dishes = 0

        # self.table_history_with_user = []
        self.dish_history_with_user = []
        self.time_previous_user_event = []
        # self.total_tables_per_user = []
        self.total_dishes_per_user = []
        self.dish_cache = {}
        self.time_kernel_prior = {}
        self.time_history_per_user = {}
        self.doc_history_per_user = {}
        self.question_history_per_user = {}
        # self.table_history_per_user = {}
        self.dish_history_per_user = {}
        self.alpha_history = {}
        self.alpha_distribution_history = {}
        self.mu_rate = mu_rate
        self.mu_per_user = {}
        self.time_elapsed = 0
        # self.active_tables_per_user = {}
        ####
        self.user_dish_cache = defaultdict(dict)
        self.author_index = author_index
        self.dish_counters_endo_per_user = defaultdict(dict)
        self.dish_counters_exo = {}

    def reseed(self, seed=None, uid=None):
        self.seed = seed
        self.prng = RandomState(self.seed)
        if uid is None:
            self.uid = self.prng.randint(maxint)
        else:
            self.uid = uid

    def reset_weight(self):
        self.logweight = 0

    def copy(self):
        new_p = Particle(users=self.users,
                         vocabulary_length=self.vocabulary_length,
                         seed=self.seed, mu_rate=self.mu_rate,
                         theta_0=self.theta_0,
                         omega=self.omega,
                         beta=self.beta,
                         mu_0=self.mu_0,
                         uid=self.uid,
                         logweight=self.logweight,
                         update_kernels=self.update_kernels,
                         keep_alpha_history=self.keep_alpha_history,
                         author_index=self.author_index)

        ####
        new_p.user_dish_cache = copy_dict(self.user_dish_cache)
        new_p.dish_counters_endo_per_user = copy_dict(self.dish_counters_endo_per_user)
        new_p.dish_counters_exo = copy_dict(self.dish_counters_exo)
        ####
        new_p.alpha_0 = copy(self.alpha_0)
        new_p.num_events = self.num_events
        new_p.topic_previous_event = self.topic_previous_event
        # new_p.total_tables = self.total_tables
        new_p.total_dishes = self.total_dishes
        new_p._max_dish = self._max_dish

        new_p.time_previous_user_event = copy(self.time_previous_user_event)
        # new_p.total_tables_per_user = copy(self.total_tables_per_user)
        new_p.total_dishes_per_user = copy(self.total_dishes_per_user)
        new_p.first_observed_time = copy(self.first_observed_time)
        new_p.first_observed_user_time = copy(self.first_observed_user_time)
        # new_p.table_history_with_user = copy(self.table_history_with_user)
        new_p.dish_history_with_user = copy(self.dish_history_with_user)

        new_p.dish_cache = copy_dict(self.dish_cache)
        new_p.dish_counters = copy_dict(self.dish_counters)

        new_p.per_topic_word_count = {}
        for doc_type in self.per_topic_word_counts:
            new_p.per_topic_word_counts[doc_type] = copy_dict(self.per_topic_word_counts[doc_type])

        new_p.per_topic_word_count_total = {}
        for doc_type in self.per_topic_word_count_total:
            new_p.per_topic_word_count_total[doc_type] = copy_dict(self.per_topic_word_count_total[doc_type])

        new_p.time_kernels = copy_dict(self.time_kernels)
        new_p.time_kernel_prior = copy_dict(self.time_kernel_prior)
        # new_p.user_table_cache = copy_dict(self.user_table_cache)
        if self.keep_alpha_history:
            new_p.alpha_history = copy_dict(self.alpha_history)
            new_p.alpha_distribution_history = \
                copy_dict(self.alpha_distribution_history)
        new_p.mu_per_user = copy_dict(self.mu_per_user)
        # new_p.active_tables_per_user = copy_dict(self.active_tables_per_user)
        return new_p

    def update(self, event):
        """Parses an event and updates the particle


        Parameters
        ----------
        event : tuple
            The event is a 4-tuple of the form (user, time, content, metadata)
        """
        # u_n : user of the n-th event
        # t_n : time of the n-th event
        # d_n : text of the n-th event
        # q_n : any metadata for the n-th event, e.g. the question id
        t_n, d_n, u_n, q_n = event
        cousers = u_n[:]
        u_n = u_n[self.author_index]
        del cousers[self.author_index]


        d_n = {vocab_type: d_n[vocab_type].split() for vocab_type in d_n}

        if self.num_events == 0:
            self.time_previous_user_event = [0 for i in range(self.max_users_id)]
            self.total_dishes_per_user = [0 for i in range(self.max_users_id)]
            self.dish_counters_endo_per_user = {i: {} for i in range(self.max_users_id)}
            self.mu_per_user = {i: self.sample_mu()
                                for i in self.users}

        if self.num_events >= 1 and u_n in self.time_previous_user_event and \
                        self.time_previous_user_event[u_n] > 0:
            log_likelihood_tn = self.time_event_log_likelihood(t_n, u_n)
        else:
            log_likelihood_tn = 0

        # tables_before = self.total_tables_per_user[u_n]
        dishes_before = self.total_dishes_per_user[u_n]

        z_n, is_exo_even, is_new_dish, log_likelihood_dn = \
            self.sample_task(t_n, d_n, u_n, cousers)

        if self.total_dishes_per_user[u_n] > dishes_before and dishes_before > 0:
            # opened a new table
            old_mu = self.mu_per_user[u_n]
            dishes_num = dishes_before + 1
            user_alive_time = t_n - self.first_observed_user_time[u_n]
            new_mu = (self.mu_rate * old_mu +
                      (1 - self.mu_rate) * dishes_num / user_alive_time)
            self.mu_per_user[u_n] = new_mu

        if z_n not in self.time_kernels:
            self.time_kernels[z_n] = self.sample_time_kernel()
            self.first_observed_time[z_n] = t_n
            self.dish_cache[z_n] = (t_n, 0, 1, 1, 1)
            self._max_dish = z_n
        else:
            if self.update_kernels:
                self.update_time_kernel(t_n, z_n)

        if self.update_kernels and self.keep_alpha_history:
            if z_n not in self.alpha_history:
                self.alpha_history[z_n] = []
                self.alpha_distribution_history[z_n] = []
            self.alpha_history[z_n].append(self.time_kernels[z_n])
            self.alpha_distribution_history[z_n].append(self.time_kernel_prior[z_n])

        if self.num_events >= 1:
            self.logweight += log_likelihood_tn
            self.logweight += self._Qn

        self.num_events += 1
        self._update_word_counters(d_n, z_n)

        self.time_previous_user_event[u_n] = t_n
        self.topic_previous_event = z_n
        self.user_previous_event = u_n

        if is_new_dish:
            self.dish_counters_exo[z_n] = 1
        elif is_exo_even:
            self.dish_counters_exo[z_n] += 1
        else:
            if z_n not in self.dish_counters_endo_per_user[u_n]:
                self.dish_counters_endo_per_user[u_n][z_n] = 1
            else:
                self.dish_counters_endo_per_user[u_n][z_n] += 1


        if u_n not in self.first_observed_user_time:
            self.first_observed_user_time[u_n] = t_n

        for c in cousers:
            self.time_previous_user_event[c] = t_n

            if z_n in self.dish_counters_endo_per_user[c]:
                self.dish_counters_endo_per_user[c][z_n] += 1
            else:
                self.dish_counters_endo_per_user[c][z_n] = 0

            #TODO: I'm not sure about this part!
            if c not in self.first_observed_user_time:
                self.first_observed_user_time[c] = t_n

        return z_n

    def sample_task(self, t_n, d_n, u_n, cousers):
        """Samples table b_n and topic z_n together for the event n.


        Parameters
        ----------
        t_n : float
            The time of the event.

        d_n : list
            The document for the event.

        u_n : int
            The user id.


        Returns
        -------
        table : int

        dish : int
        """
        # if self.total_dishes_per_user[u_n] == 0:
            # This is going to be the user's first dish
            # self.time_previous_user_event[u_n] = 0

        num_dishes = len(self.dish_counters_exo)
        intensities = []

        dn_word_counts = {vocab_type: Counter(d_n[vocab_type]) for vocab_type in d_n}
        count_dn = {vocab_type: len(d_n[vocab_type]) for vocab_type in d_n}

        # Precompute the doc_log_likelihood for each of the dishes
        dish_log_likelihood = []
        for dish in self.dish_counters_exo:
            temp_dll = {}
            for vocab_type in d_n:
                temp_dll[vocab_type] = self.document_log_likelihood(dn_word_counts[vocab_type], count_dn[vocab_type],
                                                                    dish, vocab_type)
            dll = sum(temp_dll.values())

            dish_log_likelihood.append(dll)


        # Provide one option for each of the user's dishes
        mu = self.mu_per_user[u_n]
        total_dish_intensity = mu
        dish_log_likelihood_array = []

        user_dishes = self.dish_counters_endo_per_user[u_n].keys()

        for dish in user_dishes:
            alpha = self.time_kernels[dish]
            t_last, sum_kernels = self.user_dish_cache[u_n][dish]
            update_value = self.kernel(t_n, t_last)
            dish_intensity = alpha * sum_kernels * update_value
            dish_intensity += alpha * update_value
            total_dish_intensity += dish_intensity

        log_intensities = [ln(inten_i / total_dish_intensity) + dish_log_likelihood_array[i]
                           if inten_i > 0 else -float('inf')
                           for i, inten_i in enumerate(intensities)]

        # Provide one option for new table with already existing dish
        global_dishes = self.dish_counters_exo.keys()

        for dish in global_dishes:
            # if dish in self.dish_counters_endo_per_user[u_n]:
            #     continue
            #TODO I'm not sure about the formula
            dish_intensity = (mu / total_dish_intensity) * self.dish_counters_exo[dish] / (self.total_dishes + self.beta)
            dish_intensity = ln(dish_intensity)
            dish_intensity += dish_log_likelihood[dish]
            log_intensities.append(dish_intensity)

        # Provide a last option for new table with new dish
        # new_dish_intensity = mu * self.beta / (total_table_int * (self.total_tables + self.beta))
        new_dish_intensity = mu * self.beta / (total_dish_intensity * (self.total_dishes + self.beta))

        new_dish_intensity = ln(new_dish_intensity)
        temp_dll = {}
        for vocab_type in d_n:
            temp_dll[vocab_type] = self.document_log_likelihood(dn_word_counts[vocab_type],
                                                                count_dn[vocab_type],
                                                                num_dishes, vocab_type)
        new_dish_log_likelihood = sum(temp_dll.values())

        new_dish_intensity += new_dish_log_likelihood
        log_intensities.append(new_dish_intensity)

        normalizing_log_intensity = logsumexp(log_intensities)
        intensities = [exp(log_intensity - normalizing_log_intensity)
                       for log_intensity in log_intensities]
        self._Qn = normalizing_log_intensity
        k = weighted_choice(intensities, self.prng)

        is_new_dish = False
        is_exo_even = False


        if k < len(user_dishes):
            # Assign to one of the already existing tables
            dish = user_dishes[k]
            t_last, sum_kernels = self.user_dish_cache[u_n][dish]
            update_value = self.kernel(t_n, t_last)
            sum_kernels += 1
            sum_kernels *= update_value
            self.user_dish_cache[u_n][dish] = (t_n, sum_kernels)

        else:
            is_exo_even = True

            self.total_dishes_per_user[u_n] += 1

            if k == (len(intensities) - 1):
                is_new_dish = True
                dish = self.total_dishes
                self.total_dishes += 1
            else:
                k = k - len(user_dishes)
                dish = global_dishes[k]
            
            if u_n not in self.user_dish_cache:
                self.user_dish_cache[u_n] = {}
            self.user_dish_cache[u_n][dish] = (t_n, 0)

            if dish not in self.time_kernel_prior:
                self.time_kernel_prior[dish] = self.alpha_0

                temp_dll = {}
                for doc_type in d_n:
                    temp_dll[doc_type] = self.document_log_likelihood(dn_word_counts[doc_type], count_dn[doc_type],
                                                                      dish, doc_type)
                dll = sum(temp_dll.values())
                dish_log_likelihood.append(dll)

        for c in cousers:

            if c not in self.user_dish_cache:
                self.user_dish_cache[c] = {}

            if dish not in self.user_dish_cache[c]:
                self.user_dish_cache[c][dish] = (t_n, 0)
            else:
                t_last, sum_kernels = self.user_dish_cache[c][dish]
                update_value = self.kernel(t_n, t_last)
                sum_kernels += 1
                sum_kernels *= update_value
                self.user_dish_cache[c][dish] = (t_n, sum_kernels)

        self.dish_history_with_user.append((u_n, dish))
        self.time_previous_user_event[u_n] = t_n
        return dish, is_exo_even, is_new_dish, dish_log_likelihood[dish]

    def kernel(self, t_i, t_j):
        """Returns the kernel function for t_i and t_j.


        Parameters
        ----------
        t_i : float
            The later timestamp

        t_j : float
            The earlier timestamp


        Returns
        -------
        float
        """
        return exp(-self.omega * (t_i - t_j))

    def update_time_kernel(self, t_n, z_n):
        """Updates the parameter of the time kernel of the chosen pattern
        """
        t_last, sum_kernels, event_count, intensity, prod = self.dish_cache[z_n]
        update_value = self.kernel(t_n, t_last)

        sum_kernels += 1
        sum_kernels *= update_value
        prod = sum_kernels
        sum_integrals = event_count - sum_kernels
        sum_integrals /= self.omega

        self.time_kernel_prior[z_n] = self.alpha_0[0] + event_count - self.dish_counters_exo[z_n], \
                                      self.alpha_0[1] + (sum_integrals)

        prior = self.time_kernel_prior[z_n]
        self.time_kernels[z_n] = self.sample_time_kernel(prior)

        self.dish_cache[z_n] = t_n, sum_kernels, event_count + 1, intensity, prod

    def sample_time_kernel(self, alpha_0=None):
        if alpha_0 is None:
            alpha_0 = self.alpha_0

        return self.prng.gamma(alpha_0[0], 1.0 / alpha_0[1])

    def sample_mu(self):
        """Samples a value from the prior of the base intensity mu.


        Returns
        -------
        mu_u : float
            The base intensity of a user, sampled from the prior.
        """
        return self.prng.gamma(self.mu_0[0], self.mu_0[1])

    def document_log_likelihood(self, dn_word_counts, count_dn, z_n, vocab_type):
        """Returns the log likelihood of document d_n to belong to cluster z_n.

        Note: Assumes a Gamma prior on the word distribution.
        """
        theta = self.theta_0[vocab_type][0]
        V = self.vocabulary_length[vocab_type]
        if z_n not in self.per_topic_word_count_total[vocab_type]:
            count_zn_no_dn = 0
        else:
            count_zn_no_dn = self.per_topic_word_count_total[vocab_type][z_n]
        # TODO: The code below works only for uniform theta_0. We should
        # put the theta that corresponds to `word`. Here we assume that
        # all the elements of theta_0 are equal
        gamma_numerator = _gammaln(count_zn_no_dn + V * theta)
        gamma_denominator = _gammaln(count_zn_no_dn + count_dn + V * theta)
        is_old_topic = z_n <= self._max_dish
        unique_words = len(dn_word_counts) == count_dn
        topic_words = None
        if is_old_topic:
            topic_words = self.per_topic_word_counts[vocab_type][z_n]

        if unique_words:
            rest = [_ln(topic_words[word] + theta)
                    if is_old_topic and word in topic_words
                    else self._lntheta[vocab_type]
                    for word in dn_word_counts]
        else:
            rest = [_gammaln(topic_words[word] + dn_word_counts[word] + theta) - _gammaln(topic_words[word] + theta)
                    if is_old_topic and word in topic_words
                    else _gammaln(dn_word_counts[word] + theta) - _gammaln(theta)
                    for word in dn_word_counts]
        return gamma_numerator - gamma_denominator + sum(rest)

    def document_history_log_likelihood(self):
        """Computes the log likelihood for the whole history of documents,
        using the inferred parameters.
        """
        doc_log_likelihood = 0
        for user in self.doc_history_per_user:
            for doc, table in zip(self.doc_history_per_user[user],
                                  self.table_history_per_user[user]):
                dish = self.dish_on_table_per_user[user][table]
                doc_word_counts = Counter(doc.split())
                count_doc = len(doc.split())
                doc_log_likelihood += self.document_log_likelihood(doc_word_counts,
                                                                   count_doc,
                                                                   dish)
        return doc_log_likelihood

    def time_event_log_likelihood(self, t_n, u_n):
        mu = self.mu_per_user[u_n]
        integral = (t_n - self.time_previous_user_event[u_n]) * mu
        intensity = mu

        if u_n in self.user_dish_cache:
            for dish in self.user_dish_cache[u_n]:
                t_last, sum_timedeltas = self.user_dish_cache[u_n][dish]
                update_value = self.kernel(t_n, t_last)
                topic_sum = (sum_timedeltas + 1) - \
                            (sum_timedeltas + 1) * update_value
                topic_sum *= self.time_kernels[dish]
                integral += topic_sum
                intensity += (sum_timedeltas + 1) \
                             * self.time_kernels[dish] * update_value

        return ln(intensity) - integral

    def _update_word_counters(self, d_n, z_n):

        for doc_type in d_n:
            if z_n not in self.per_topic_word_counts[doc_type]:
                self.per_topic_word_counts[doc_type][z_n] = {}
            if z_n not in self.per_topic_word_count_total[doc_type]:
                self.per_topic_word_count_total[doc_type][z_n] = 0
            for word in d_n[doc_type]:
                if word not in self.per_topic_word_counts[doc_type][z_n]:
                    self.per_topic_word_counts[doc_type][z_n][word] = 0
                self.per_topic_word_counts[doc_type][z_n][word] += 1
                self.per_topic_word_count_total[doc_type][z_n] += 1
        return

    def to_process(self):
        """Exports the particle as a HDHProcess object.

        Use the exported object to plot the user timelines.

        Returns
        -------
        HDHProcess
        """
        process = HDHProcess(num_patterns=len(self.time_kernels),
                             mu_0=self.mu_0,
                             alpha_0=self.alpha_0,
                             vocabulary=self.vocabulary, num_users=self.num_users,
                             vocab_types=[vocab_type for vocab_type in self.vocabulary_length])
        process.mu_per_user = self.mu_per_user
        # process.table_history_per_user = self.table_history_per_user
        process.dish_history_per_user = self.dish_history_per_user
        process.time_history_per_user = self.time_history_per_user
        process.dish_on_table_per_user = self.dish_on_table_per_user
        process.time_kernels = self.time_kernels
        process.first_observed_time = self.first_observed_time
        process.omega = self.omega
        process.num_users = self.num_users
        process.document_history_per_user = self.doc_history_per_user
        process.per_pattern_word_counts = copy_dict(self.per_topic_word_counts)
        process.per_pattern_word_count_total = copy_dict(self.per_topic_word_count_total)
        process.theta_0 = copy_dict(self.theta_0)
        process.vocabulary_length = self.vocabulary_length
        process._max_dish = self._max_dish
        process._lntheta = self._lntheta

        return process

    # def get_intensity(self, t_n, u_n, z_n):
    #     pi_z = self.dish_counters[z_n] / self.total_tables
    #     mu = self.mu_per_user[u_n]
    #     alpha = self.time_kernels[z_n]
    #     intensity = pi_z * mu
    #     for table in self.user_table_cache[u_n]:
    #         dish = self.dish_on_table_per_user[u_n][table]
    #         if dish == z_n:
    #             t_last, sum_timedeltas = self.user_table_cache[u_n][table]
    #             update_value = self.kernel(t_n, t_last)
    #             table_intensity = alpha * sum_timedeltas * update_value
    #             table_intensity += alpha * update_value
    #             intensity += table_intensity
    #     return intensity


def _extract_words_users(history, vocab_types):
    """Returns the set of words and the set of users in the dataset
    """
    vocabulary = {vocab_type: set() for vocab_type in vocab_types}
    users = set()

    for t, doc, all_users, q in history:
        for u in all_users:
            users.add(u)
        for doc_type in doc:
            for word in doc[doc_type].split():
                vocabulary[doc_type].add(word)
    vocabulary = {vocab_type: list(vocabulary[vocab_type]) for vocab_type in vocabulary}

    return vocabulary, users


def resample_indices(weights, prng):
    N = len(weights)
    index = prng.randint(N)
    beta = 0.0
    mw = max(weights)
    picked_indices = []
    for i in range(N):
        beta += prng.rand() * 2.0 * mw
        while beta > weights[index]:
            beta -= weights[index]
            index = (index + 1) % N
        picked_indices.append(index)
    return sorted(picked_indices)


def pick_new_particles(old_particles, weights, prng):
    N = len(old_particles)
    index = prng.randint(N)
    beta = 0.0
    mw = max(weights)
    picked_indices = []
    for i in range(N):
        beta += prng.rand() * 2.0 * mw
        while beta > weights[index]:
            beta -= weights[index]
            index = (index + 1) % N
        picked_indices.append(index)
    return picked_indices


def _infer_single_thread(history, params):
    prng = RandomState(seed=params.seed)
    time_history_per_user = defaultdict(list)
    doc_history_per_user = defaultdict(list)
    question_history_per_user = defaultdict(list)
    # table_history_with_user = []
    # dish_on_table_per_user = []

    # Set the accuracy
    count_resamples = 0
    square_norms = []
    with open(params.progress_file, 'a') as out:
        out.write('Starting %d particles on %d thread.\n' % (params.num_particles,
                                                             params.threads))

    start_tic = time()

    # Initialize the particles
    epsilon = 1e-10
    particles = [Particle(theta_0=params.theta_0, alpha_0=params.alpha_0,
                          mu_0=params.mu_0,
                          uid=prng.randint(maxint), seed=prng.randint(maxint),
                          vocabulary_length={vocab_type: len(params.vocabulary[vocab_type]) for vocab_type in
                                             params.vocabulary},
                          update_kernels=params.update_kernels,
                          omega=params.omega, beta=params.beta,
                          users=params.users,
                          keep_alpha_history=params.keep_alpha_history,
                          mu_rate=params.mu_rate,
                          author_index=params.author_index)
                 for i in range(params.num_particles)]

    # inferred_tables = {}  # for each particle, save the topic history
    inferred_dishes = {}  # for each particle, save the topic history
    for p in particles:
        inferred_dishes[p.uid] = []
    # Fit each particle to the history
    square_norms = []
    # table_history_with_user = []
    dish_history_with_user = []
    # dish_on_table_per_user = []
    for i, h_i in enumerate(history):
        max_logweight = None
        weights = []
        total = 0
        t_i, d_i, u_i, q_i = h_i
        u_i = u_i[params.author_index]

        if u_i not in time_history_per_user:
            time_history_per_user[u_i] = []
            doc_history_per_user[u_i] = []
            question_history_per_user[u_i] = []
        time_history_per_user[u_i].append(t_i)
        doc_history_per_user[u_i].append(d_i)
        question_history_per_user[u_i].append(q_i)

        for p_i in particles:
            # Fit each particle to the next event
            z_i = p_i.update(h_i)
            # inferred_tables[p_i.uid].append((b_i, z_i))
            inferred_dishes[p_i.uid].append(z_i)

        if i > 0 and i % params.resample_every == 0:
            # Start resampling
            for p_i in particles:
                if max_logweight is None or max_logweight < p_i.logweight:
                    max_logweight = p_i.logweight
            for p_i in particles:
                # Normalize the weights of the  particles
                if p_i.logweight - max_logweight >= \
                                ln(epsilon) - ln(params.num_particles):
                    weights.append(exp(p_i.logweight - max_logweight))
                else:
                    weights.append(exp(p_i.logweight - max_logweight))
                total += weights[-1]
            normalized = [w / sum(weights) for w in weights]
            # Check if resampling is needed
            norm2 = sum([w ** 2 for w in normalized])
            square_norms.append(norm2)
            if params.num_particles > 1 \
                    and norm2 > params.particle_weight_threshold / params.num_particles \
                    and i < len(history) - 1:
                # Resample particles (though never for the last event)
                count_resamples += 1
                new_particle_indices = pick_new_particles(particles,
                                                          normalized, prng)
                new_particles = []
                new_dish_history_with_user = []
                # new_dish_on_table_per_user = []

                for index in new_particle_indices:
                    # copy table_history for that particle
                    if len(dish_history_with_user):
                        old_history = copy(dish_history_with_user[index])
                    else:
                        old_history = []
                    new_history = copy(particles[index].dish_history_with_user)
                    old_history.extend(new_history)
                    new_dish_history_with_user.append(old_history)
                    # if len(dish_on_table_per_user):
                    #     dish_table_user = copy_dict(dish_on_table_per_user[index])
                    # else:
                    #     dish_table_user = {}
                    # dishes_toadd = copy_dict(particles[index].dish_on_table_todelete)
                    # for user in dishes_toadd:
                    #     if user not in dish_table_user:
                    #         dish_table_user[user] = {}
                    #     for t in dishes_toadd[user]:
                    #         assert t not in dish_table_user[user]
                    #         dish_table_user[user][t] = dishes_toadd[user][t]
                    # new_dish_on_table_per_user.append(dish_table_user)

                # delete history from new particles
                for index in new_particle_indices:
                    particles[index].dish_history_with_user = []
                    # for user in particles[index].dish_on_table_todelete:
                    #     particles[index].dish_on_table_todelete[user] = {}

                for index in new_particle_indices:
                    particles[index].dish_history_with_user = []
                    new_particle = particles[index].copy()
                    new_particle.reseed(prng.randint(maxint))
                    new_particle.reset_weight()
                    new_particles.append(new_particle)
                    inferred_dishes[new_particle.uid] = \
                        copy(inferred_dishes[particles[index].uid])
                particles = new_particles
                dish_history_with_user = new_dish_history_with_user
                # dish_on_table_per_user = new_dish_on_table_per_user

                # If inferred tables dictionary grows too big, prune it
                if len(inferred_dishes) > 50 * params.num_particles:
                    new_inferred_dishes = {}
                    for p in particles:
                        new_inferred_dishes[p.uid] = copy(inferred_dishes[p.uid])
                    del inferred_dishes
                    inferred_dishes = new_inferred_dishes
                with open(params.progress_file, mode='a') as temp:
                    temp.write("Time: %.2f (%d)\n" % (time() - start_tic, i))

    # Finally sample a single particle according to its weight.
    for p_i in particles:
        if max_logweight is None or max_logweight < p_i.logweight:
            max_logweight = p_i.logweight
    for p_i in particles:
        # Normalize the weights of the  particles
        if p_i.logweight - max_logweight >= \
                        ln(epsilon) - ln(params.num_particles):
            weights.append(exp(p_i.logweight - max_logweight))
        else:
            weights.append(exp(p_i.logweight - max_logweight))
        total += weights[-1]
    normalized = [w / sum(weights) for w in weights]
    final_particle_id = pick_new_particles(particles, normalized, prng)[0]
    final_particle = particles[final_particle_id]

    dish_history_with_user = dish_history_with_user[final_particle_id]
    new_history = copy(final_particle.dish_history_with_user)
    dish_history_with_user.extend(new_history)
    final_particle.dish_history_with_user = dish_history_with_user
    # dish_on_table_per_user = dish_on_table_per_user[final_particle_id]
    # dishes_toadd = copy_dict(final_particle.dish_on_table_per_user)

    # for user in dishes_toadd:
    #     if user not in dish_on_table_per_user:
    #         dish_on_table_per_user[user] = {}
    #     for t in dishes_toadd[user]:
    #         assert t not in dish_on_table_per_user[user]
    #         dish_on_table_per_user[user][t] = dishes_toadd[user][t]
    # for user in final_particle.dish_on_table_todelete:
    #     if user not in dish_on_table_per_user:
    #         dish_on_table_per_user[user] = {}
    #     for t in final_particle.dish_on_table_todelete[user]:
    #         assert t not in dish_on_table_per_user[user]
    #         dish_on_table_per_user[user][t] = \
    #             final_particle.dish_on_table_todelete[user][t]
    # final_particle.dish_on_table_per_user = dish_on_table_per_user

    final_particle.time_history_per_user = copy(time_history_per_user)
    final_particle.doc_history_per_user = copy(doc_history_per_user)
    final_particle.question_history_per_user = copy(question_history_per_user)
    final_particle.dish_history_per_user = {}

    for (u_i, dish) in final_particle.dish_history_with_user:
        if u_i not in final_particle.dish_history_per_user:
            final_particle.dish_history_per_user[u_i] = []
        final_particle.dish_history_per_user[u_i].append(dish)
    final_particle.vocabulary = params.vocabulary
    # pool.close()
    with open(params.progress_file, mode='a') as temp:
        temp.write("Resampled %d times\n" % (count_resamples))
        temp.write("Finished in time: %.2f\n" %
                   (time() - start_tic))
    return final_particle, square_norms


def infer(history, alpha_0, mu_0, vocab_types, omega=1, beta=1, theta_0=None,
          threads=1, num_particles=1,
          particle_weight_threshold=1, resample_every=10,
          update_kernels=True, mu_rate=0.6,
          # enable_log=False, logfile='particles.log',
          keep_alpha_history=False, progress_file=None, seed=None, author_index=0):
    """Runs the inference algorithm and returns a particle.

    Parameters
    ----------
    history : list
        A list of 4-tuples (user, time, content, metadata) that represents the
        event history that we want to infer our model on.

    alpha_0 : tuple
        The Gamma prior parameter for a pattern's time kernel.

    mu_0 : tuple
        The Gamma prior parameter for the user activity rate.

    omega : float
        The time decay parameter.

    beta : float
        A parameter that controls the new-task probability.

    theta_0 :  list, default is None
        If not None, theta_0 corresponds to the Dirichlet prior used for the
        word distribution. It should have as many dimensions as the number of
        words. By default, this is the vector :math:`[1 / |V|, \ldots, 1 / |V|]`, where
        :math:`|V|` is the size of the vocabulary.

    threads : int, default is 1
        The number of CPU threads that will be used during inference.

    num_particles : int, default is 1
        The number of particles that the SMC algorithm will use.

    particle_weight_threshold : float, default is 1
        A parameter that controls when the particles need to be re-sampled

    re-sample_every : int, default is 10
        The frequency with which we check if we need to re-sample or not. The
        number is in inference steps (number of events)

    update_kernels : bool, default is True
        Controls wheter the time kernel parameter of each pattern will be
        updated from the posterior, or not.

    mu_rate : float, default is 0.6
        The learning-rate with which we update the activity rate of
        a user.

    keep_alpha_history : bool, default is False
        For debug reasons, we make want to keep the complete history of the value
        of each pattern's time kernel parameter as we see more events in that
        pattern.

    progress_file : str, default is None
        Since the computation might be slow, we want to save progress
        information to a file instead of printing it. If None, a temporary,
        randomly-named file is generated for this purpose.
    """

    vocabulary, users = _extract_words_users(history, vocab_types)

    if theta_0 is None:
        theta_0 = {vocab_type: [] for vocab_type in vocab_types}
        for vocab_type in theta_0:
            theta_0[vocab_type] = [1 / len(vocabulary[vocab_type])] * len(vocabulary[vocab_type])

    if progress_file is None:
        with tempfile.NamedTemporaryFile(mode='a', suffix='.log', dir='.',
                                         delete=False) as temp:
            progress_file = temp.name
            print('Created temporary log file %s' % (progress_file))
    params = InferenceParameters(alpha_0, mu_0, omega, beta, theta_0, threads,
                                 num_particles, particle_weight_threshold,
                                 resample_every, update_kernels, mu_rate,
                                 keep_alpha_history, progress_file, seed,
                                 vocabulary, users, author_index)

    if threads == 1:
        return _infer_single_thread(history, params)
    else:
        raise NotImplementedError("Multi-threaded version not yet implemented")
