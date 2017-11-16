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
from copy import copy, deepcopy
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
                 vocabulary, users):
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


class Particle(object):
    def __init__(self, vocabulary_length, num_users, time_kernels=None,
                 alpha_0=(2, 2), mu_0 = 1, theta_0=None,
                 seed=None, logweight=0, update_kernels=False, uid=0,
                 omega=1, beta=1, keep_alpha_history=False, mu_rate=0.6):
        self.count = 0 # modified
        self.vocabulary_length = vocabulary_length
        self.vocabulary = None
        self.seed = seed
        self.prng = RandomState(self.seed)
        self.first_observed_time = {}
        self.first_observed_user_time = {}
        # modified (extend to multiple document types)
        self.per_topic_word_counts = {k: dict () for k in vocabulary_length}
        self.per_topic_word_count_total = {k: dict () for k in vocabulary_length}

        self.time_kernels = {}
        self.alpha_0 = alpha_0
        self.mu_0 = mu_0
        # modified (extend to multiple document types)
        self.theta_0 = {k:array(theta_0[k]) for k in theta_0}
        self._lntheta = {k: _ln(theta_0[k][0]) for k in self.theta_0}

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
        self.num_users = num_users
        self.keep_alpha_history = keep_alpha_history

        self.user_table_cache = {}
        self.dish_on_table_per_user = {}
        self.dish_on_table_todelete = {}
        self.dish_counters = {}
        self._max_dish = -1
        self.total_tables = 0

        self.table_history_with_user = []
        self.time_previous_user_event = []
        self.total_tables_per_user = []
        self.dish_cache = {}
        self.time_kernel_prior = {}
        self.time_history_per_user = {}
        self.doc_history_per_user = {}
        self.question_history_per_user = {}
        self.table_history_per_user = {}
        self.alpha_history = {}
        self.alpha_distribution_history = {}
        self.mu_rate = mu_rate
        self.mu_per_user = {}
        self.time_elapsed = 0
        self.active_tables_per_user = {}

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
        new_p = Particle(num_users=self.num_users,
                         vocabulary_length=self.vocabulary_length,
                         seed=self.seed, mu_rate=self.mu_rate,
                         theta_0=self.theta_0,
                         omega=self.omega,
                         beta=self.beta,
                         mu_0=self.mu_0,
                         uid=self.uid,
                         logweight=self.logweight,
                         update_kernels=self.update_kernels,
                         keep_alpha_history=self.keep_alpha_history)
        new_p.alpha_0 = copy(self.alpha_0)
        new_p.num_events = self.num_events
        new_p.topic_previous_event = self.topic_previous_event
        new_p.total_tables = self.total_tables
        new_p._max_dish = self._max_dish

        new_p.time_previous_user_event = copy(self.time_previous_user_event)
        new_p.total_tables_per_user = copy(self.total_tables_per_user)
        new_p.first_observed_time = copy(self.first_observed_time)
        new_p.first_observed_user_time = copy(self.first_observed_user_time)
        new_p.table_history_with_user = copy(self.table_history_with_user)

        new_p.dish_cache = copy_dict(self.dish_cache)
        new_p.dish_counters = copy_dict(self.dish_counters)
        new_p.dish_on_table_per_user = \
            copy_dict(self.dish_on_table_per_user)

        new_p.dish_on_table_per_user = {}
        new_p.dish_on_table_todelete = {}
        for u in self.dish_on_table_per_user:
            new_p.dish_on_table_per_user[u] = {}
            new_p.dish_on_table_todelete[u] = {}
            self.dish_on_table_todelete[u] = {}

            for t in self.dish_on_table_per_user[u]:
                if t in self.active_tables_per_user[u]:
                    new_p.dish_on_table_per_user[u][t] = \
                        self.dish_on_table_per_user[u][t]
                else:
                    dish = self.dish_on_table_per_user[u][t]
                    self.dish_on_table_todelete[u][t] = dish
                    new_p.dish_on_table_todelete[u][t] = dish
                    if t in self.user_table_cache[u]:
                        del self.user_table_cache[u][t]

        # Modified (because I changed to data structures I simply do a deep copy)
        new_p.per_topic_word_counts = deepcopy(self.per_topic_word_counts)
        new_p.per_topic_word_count_total = deepcopy(self.per_topic_word_count_total)
        new_p.time_kernels = copy_dict(self.time_kernels)
        new_p.time_kernel_prior = copy_dict(self.time_kernel_prior)
        new_p.user_table_cache = copy_dict(self.user_table_cache)
        if self.keep_alpha_history:
            new_p.alpha_history = copy_dict(self.alpha_history)
            new_p.alpha_distribution_history = \
                copy_dict(self.alpha_distribution_history)
        new_p.mu_per_user = copy_dict(self.mu_per_user)
        new_p.active_tables_per_user = copy_dict(self.active_tables_per_user)
        return new_p

    def update(self, event):
        """Parses an event and updates the particle


        Parameters
        ----------
        event : tuple
            The event is a 4-tuple of the form (time, content, user, metadata)
        """
        # t_n : time of the n-th event
        # d_n : text of the n-th event
        # u_n : user of the n-th event
        # q_n : any metadata for the n-th event, e.g. the question id
        t_n, d_n, u_n, q_n = event
        # modified (u_n is a list and the first item in the list is the leading author)
        cousers = u_n[1:]
        u_n = u_n[0]
        #d_n = d_n.split()

        # the following conditional block can be removed and added to the initialization code.

        if self.num_events == 0:
            self.time_previous_user_event = [0 for i in range(self.num_users)]
            self.total_tables_per_user = [0 for i in range(self.num_users)]
            self.mu_per_user = {i: self.sample_mu()
                                for i in range(self.num_users)}
            self.active_tables_per_user = {i: set()
                                           for i in range(self.num_users)}
        if self.num_events >= 1 and u_n in self.time_previous_user_event and \
                self.time_previous_user_event[u_n] > 0:
            log_likelihood_tn = self.time_event_log_likelihood(t_n, u_n)
        else:
            log_likelihood_tn = 0 # I don't know why this is zero, should it not be the log of the base intensity?

        tables_before = self.total_tables_per_user[u_n]
        b_n, z_n, opened_table, log_likelihood_dn = \
            self.sample_table(t_n, d_n, u_n, cousers) # modified
        if self.total_tables_per_user[u_n] > tables_before and tables_before > 0:
            # opened a new table
            old_mu = self.mu_per_user[u_n]
            tables_num = tables_before + 1
            user_alive_time = t_n - self.first_observed_user_time[u_n]
            #print (t_n, self.first_observed_user_time[u_n], user_alive_time)
            new_mu = (self.mu_rate * old_mu +
                      (1 - self.mu_rate) * tables_num / user_alive_time)
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
            self.logweight += log_likelihood_tn # this quantity changes when multiple authors are added
            self.logweight += self._Qn # this quantity changes when  multiple vocabularies are added
        self.num_events += 1
        self._update_word_counters(d_n, z_n) # modified

        self.time_previous_user_event[u_n] = t_n
        self.topic_previous_event = z_n
        self.user_previous_event = u_n
        self.table_previous_event = b_n
        self.active_tables_per_user[u_n].add(b_n)
        if z_n not in self.dish_counters:
            self.dish_counters[z_n] = 1
        elif opened_table:
            self.dish_counters[z_n] += 1
        if u_n not in self.first_observed_user_time:
            self.first_observed_user_time[u_n] = t_n
        return b_n, z_n

    def sample_table(self, t_n, d_n, u_n, cousers):
        """Samples table b_n and topic z_n together for the event n.


        Parameters
        ----------
        t_n : float
            The time of the event.

        d_n : dict
            The documents for the event. The key is the document type and the value
            is the document content as text.

        u_n : int
            The user id.


        Returns
        -------
        table : int

        dish : int

        opened_table: boolean
            True if a new table is opened, False otherwise

        dish_log_likelihood: float

        """
        # modified (update the data structures for all the co-authors)
        if self.total_tables_per_user[u_n] == 0:
            # This is going to be the user's first table
            self.dish_on_table_per_user[u_n] = {} # do we need to update this for every co-user?
            self.user_table_cache[u_n] = {}
            self.time_previous_user_event[u_n] = 0
        for c in cousers:
            if self.total_tables_per_user[c] == 0:
                # This is going to be the user's first table
                self.dish_on_table_per_user[c] = {}
                self.user_table_cache[c] = {}
                self.time_previous_user_event[c] = 0

        tables = range(self.total_tables_per_user[u_n])
        num_dishes = len(self.dish_counters)
        intensities = []
        # modified (for all document types)
        dn_word_counts = {key: Counter (d_n[key].split()) for key in d_n} # modified (for all document types)
        count_dn = {key: len (d_n[key].split()) for key in d_n}

        # Precompute the doc_log_likelihood for each of the dishes
        dish_log_likelihood = []
        for dish in self.dish_counters:
            dll = self.document_log_likelihood(dn_word_counts, count_dn,
                                               dish) # modified
            dish_log_likelihood.append(dll)

        table_intensity_threshold = 1e-8  # below this, the table is inactive

        # Provide one option for each of the already open tables
        mu = self.mu_per_user[u_n]
        total_table_int = mu
        dish_log_likelihood_array = []
        for table in tables:
            if table in self.active_tables_per_user[u_n]:
                dish = self.dish_on_table_per_user[u_n][table]
                alpha = self.time_kernels[dish]
                t_last, sum_kernels = self.user_table_cache[u_n][table]
                update_value = self.kernel(t_n, t_last)
                table_intensity = alpha * sum_kernels * update_value
                table_intensity += alpha * update_value
                total_table_int += table_intensity
                if table_intensity < table_intensity_threshold:
                    self.active_tables_per_user[u_n].remove(table)
                dish_log_likelihood_array.append(dish_log_likelihood[dish])
                intensities.append(table_intensity)
            else:
                dish_log_likelihood_array.append(0)
                intensities.append(0)
        log_intensities = [ln(inten_i / total_table_int) + sum(dish_log_likelihood_array[i].values())
                           if inten_i > 0 else -float('inf')
                           for i, inten_i in enumerate(intensities)]

        # Provide one option for new table with already existing dish
        for dish in self.dish_counters:
            dish_intensity = (mu / total_table_int) *\
                self.dish_counters[dish] / (self.total_tables + self.beta)
            dish_intensity = ln(dish_intensity)
            dish_intensity += sum(dish_log_likelihood[dish].values())
            log_intensities.append(dish_intensity)

        # Provide a last option for new table with new dish
        new_dish_intensity = mu * self.beta /\
            (total_table_int * (self.total_tables + self.beta))
        new_dish_intensity = ln(new_dish_intensity)
        new_dish_log_likelihood = self.document_log_likelihood(dn_word_counts,
                                                               count_dn,
                                                               num_dishes)
        new_dish_intensity += sum(new_dish_log_likelihood.values())
        log_intensities.append(new_dish_intensity)

        normalizing_log_intensity = logsumexp(log_intensities)
        intensities = [exp(log_intensity - normalizing_log_intensity)
                       for log_intensity in log_intensities]
        self._Qn = normalizing_log_intensity
        k = weighted_choice(intensities, self.prng)
        opened_table = False
        if k in tables:
            # Assign to one of the already existing tables
            table = k
            dish = self.dish_on_table_per_user[u_n][table]
            # update cache for that table
            t_last, sum_kernels = self.user_table_cache[u_n][table]
            update_value = self.kernel(t_n, t_last)
            sum_kernels += 1
            sum_kernels *= update_value
            self.user_table_cache[u_n][table] = (t_n, sum_kernels)
            # modified (update the history for all cousers)
            for c in cousers:
                if table not in self.user_table_cache[c]:
                    self.user_table_cache[c][table] = (t_n, 0)
                else:
                    tl, s = self.user_table_cache[c][table]
                    upval = self.kernel (t_n, tl)
                    s += 1
                    s *= upval
                    self.user_table_cache[c][table] = (t_n, s)
        else:
            k = k - len(tables)
            table = len(tables)
            self.total_tables += 1
            self.total_tables_per_user[u_n] += 1
            dish = k
            # Since this is a new table, initialize the cache accordingly
            self.user_table_cache[u_n][table] = (t_n, 0)
            # modified (update the history for all cousers)
            for c in cousers:
                if table not in self.user_table_cache[c]:
                    self.user_table_cache[c][table] = (t_n, 0)
                else:
                    tl, s = self.user_table_cache[c][table]
                    upval = self.kernel (t_n, tl)
                    s += 1
                    s *= upval
                    self.user_table_cache[c][table] = (t_n, s)

            self.dish_on_table_per_user[u_n][table] = dish
            opened_table = True
            if dish not in self.time_kernel_prior:
                self.time_kernel_prior[dish] = self.alpha_0
                dll = self.document_log_likelihood(dn_word_counts, count_dn,
                                                   dish)
                dish_log_likelihood.append(dll)

        self.table_history_with_user.append((u_n, table))
        self.time_previous_user_event[u_n] = t_n
        # modified (update the history for all cousers)
        for c in cousers:
            self.time_previous_user_event[c] = t_n
        return table, dish, opened_table, dish_log_likelihood[dish]

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
        v_1, v_2 = self.time_kernel_prior[z_n]
        t_last, sum_kernels, event_count, intensity, prod = self.dish_cache[z_n]
        update_value = self.kernel(t_n, t_last)

        sum_kernels += 1
        sum_kernels *= update_value
        prod = sum_kernels
        sum_integrals = event_count - sum_kernels
        sum_integrals /= self.omega

        self.time_kernel_prior[z_n] = self.alpha_0[0] + event_count - self.dish_counters[z_n], \
            self.alpha_0[1] + (sum_integrals)
        prior = self.time_kernel_prior[z_n]
        self.time_kernels[z_n] = self.sample_time_kernel(prior)

        self.dish_cache[z_n] = t_n, sum_kernels, event_count + 1, intensity, prod

    def sample_time_kernel(self, alpha_0=None):
        if alpha_0 is None:
            alpha_0 = self.alpha_0
        return self.prng.gamma(alpha_0[0], 1. / alpha_0[1])

    def sample_mu(self):
        """Samples a value from the prior of the base intensity mu.


        Returns
        -------
        mu_u : float
            The base intensity of a user, sampled from the prior.
        """
        return self.prng.gamma(self.mu_0[0], self.mu_0[1])

    def document_log_likelihood(self, dn_word_counts, count_dn, z_n):
        """Returns the log likelihood of document d_n to belong to cluster z_n.

        Note: Assumes a Gamma prior on the word distribution.
        """
        types = self.vocabulary_length.keys()
        dll = {dtype: 0 for dtype in types}

        for dtype in types:
            theta = self.theta_0[dtype][0]
            V = self.vocabulary_length[dtype]
            if z_n not in self.per_topic_word_count_total[dtype]:
                count_zn_no_dn = 0
            else:
                count_zn_no_dn = self.per_topic_word_count_total[dtype][z_n]
            # TODO: The code below works only for uniform theta_0. We should
            # put the theta that corresponds to `word`. Here we assume that
            # all the elements of theta_0 are equal
            gamma_numerator = _gammaln(count_zn_no_dn + V * theta)
            gamma_denominator = _gammaln(count_zn_no_dn + count_dn[dtype] + V * theta)
            is_old_topic = z_n <= self._max_dish
            unique_words = len(dn_word_counts[dtype]) == count_dn[dtype]
            topic_words = None
            if is_old_topic:
                topic_words = self.per_topic_word_counts[dtype][z_n]

            if unique_words:
                rest = [_ln(topic_words[word] + theta)
                        if is_old_topic and word in topic_words
                        else self._lntheta[dtype]
                        for word in dn_word_counts[dtype]]
            else:
                rest = [_gammaln(topic_words[word] + dn_word_counts[dtype][word] + theta) - _gammaln(topic_words[word] + theta)
                        if is_old_topic and word in topic_words
                        else _gammaln(dn_word_counts[dtype][word] + theta) - _gammaln(theta)
                        for word in dn_word_counts[dtype]]
            dll[dtype] = gamma_numerator - gamma_denominator + sum (rest)
        return dll

    def document_history_log_likelihood(self):
        """Computes the log likelihood for the whole history of documents,
        using the inferred parameters.
        """
        doc_log_likelihood = 0
        for user in self.doc_history_per_user:
            for doc, table in zip(self.doc_history_per_user[user],
                                  self.table_history_per_user[user]):
                dish = self.dish_on_table_per_user[user][table]

                dn_word_counts = {key: Counter (doc[key].split()) for key in doc}
                count_dn = {key: len (doc[key].split()) for key in doc}

                doc_log_likelihood += sum(self.document_log_likelihood(doc_word_counts,
                                                                   count_doc,
                                                                   dish).values())
        return doc_log_likelihood

    def time_event_log_likelihood(self, t_n, u_n):
        # looks like this just calculates the hawkes process likelihood (FIXME later)
        mu = self.mu_per_user[u_n]
        integral = (t_n - self.time_previous_user_event[u_n]) * mu
        intensity = mu
        for table in self.user_table_cache[u_n]:
            t_last, sum_timedeltas = self.user_table_cache[u_n][table]
            update_value = self.kernel(t_n, t_last)
            topic_sum = (sum_timedeltas + 1) - \
                (sum_timedeltas + 1) * update_value
            dish = self.dish_on_table_per_user[u_n][table]
            topic_sum *= self.time_kernels[dish]
            integral += topic_sum
            intensity += (sum_timedeltas + 1) \
                * self.time_kernels[dish] * update_value
        return ln(intensity) - integral

    def _update_word_counters(self, d_n, z_n):
        types = d_n.keys()
        for dtype in types:
            doc_length = len (d_n[dtype].split())
            if z_n not in self.per_topic_word_counts[dtype]:
                self.per_topic_word_counts[dtype][z_n] = {}
            if z_n not in self.per_topic_word_count_total[dtype]:
                self.per_topic_word_count_total[dtype][z_n] = 0
            for word in d_n[dtype].split():
                if word not in self.per_topic_word_counts[dtype][z_n]:
                    self.per_topic_word_counts[dtype][z_n][word] = 0
                self.per_topic_word_counts[dtype][z_n][word] += 1
                self.per_topic_word_count_total[dtype][z_n] += 1
        return

    def to_process(self):
        """Exports the particle as a HDHProcess object.

        Use the exported object to plot the user timelines.

        Returns
        -------
        HDHProcess
        """
        # the last two are None because they are used only for generation
        process = HDHProcess(num_users = len (self.mu_per_user),
                             num_patterns=len(self.time_kernels),
                             alpha_0=self.alpha_0,
                             mu_0=self.mu_0,
                             vocabulary=self.vocabulary,
                             omega=self.omega,
                             doc_lengths=None,
                             words_per_pattern=None,
                             cousers=None,
                             random_state=12,
                             generate=False)
        process.mu_per_user = self.mu_per_user
        process.table_history_per_user = self.table_history_per_user
        process.time_history_per_user = self.time_history_per_user
        process.dish_on_table_per_user = self.dish_on_table_per_user
        process.time_kernels = self.time_kernels
        process.first_observed_time = self.first_observed_time
        process.omega = self.omega
        process.num_users = self.num_users
        process.document_history_per_user = self.doc_history_per_user
        process.per_pattern_word_counts = copy_dict(self.per_topic_word_counts)
        process.per_pattern_word_count_total = copy_dict(self.per_topic_word_count_total)
        return process

    def get_intensity(self, t_n, u_n, z_n):
        pi_z = self.dish_counters[z_n] / self.total_tables
        mu = self.mu_per_user[u_n]
        alpha = self.time_kernels[z_n]
        intensity = pi_z * mu
        for table in self.user_table_cache[u_n]:
            dish = self.dish_on_table_per_user[u_n][table]
            if dish == z_n:
                t_last, sum_timedeltas = self.user_table_cache[u_n][table]
                update_value = self.kernel(t_n, t_last)
                table_intensity = alpha * sum_timedeltas * update_value
                table_intensity += alpha * update_value
                intensity += table_intensity
        return intensity


def _extract_words_users(history, docTypes):
    """Returns the set of words and the set of users in the dataset
    """
    vocabulary = {docType: set () for docType in docTypes}
    users = set ()

    for t, doc, u, q in history:
        for docType in docTypes:
            for word in doc[docType].split ():
                vocabulary[docType].add (word)
        for elem in u:
            users.add (elem)
    return {key: list (vocabulary[key]) for key in vocabulary}, users

def _initialize_document_distributions (vocabulary, docTypes, theta_0):
    if theta_0 is None:
        theta_0 = dict ()
        for docType in docTypes:
            theta_0[docType] = [1 / len (vocabulary[docType])] * len (vocabulary[docType])

    return theta_0


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
                          vocabulary_length={dtype: len(params.vocabulary[dtype]) for dtype in params.vocabulary},
                          update_kernels=params.update_kernels,
                          omega=params.omega, beta=params.beta,
                          num_users=len(params.users),
                          keep_alpha_history=params.keep_alpha_history,
                          mu_rate=params.mu_rate)
                 for i in range(params.num_particles)]

    inferred_tables = {}  # for each particle, save the topic history
    for p in particles:
        inferred_tables[p.uid] = []
    # Fit each particle to the history
    square_norms = []
    table_history_with_user = []
    dish_on_table_per_user = []
    for i, h_i in enumerate(history):
        max_logweight = None
        weights = []
        total = 0
        t_i, d_i, u_i, q_i = h_i
        u_i = u_i[0]
        if u_i not in time_history_per_user:
            time_history_per_user[u_i] = []
            doc_history_per_user[u_i] = []
            question_history_per_user[u_i] = []
        time_history_per_user[u_i].append(t_i)
        doc_history_per_user[u_i].append(d_i)
        question_history_per_user[u_i].append(q_i)

        for p_i in particles:
            # Fit each particle to the next event
            b_i, z_i = p_i.update(h_i)
            inferred_tables[p_i.uid].append((b_i, z_i))

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
                    and norm2 > params.particle_weight_threshold / params.num_particles\
                    and i < len(history) - 1:
                # Resample particles (though never for the last event)
                count_resamples += 1
                new_particle_indices = pick_new_particles(particles,
                                                          normalized, prng)
                new_particles = []
                new_table_history_with_user = []
                new_dish_on_table_per_user = []
                for index in new_particle_indices:
                    # copy table_history for that particle
                    if len(table_history_with_user):
                        old_history = copy(table_history_with_user[index])
                    else:
                        old_history = []
                    new_history = copy(particles[index].table_history_with_user)
                    old_history.extend(new_history)
                    new_table_history_with_user.append(old_history)
                    if len(dish_on_table_per_user):
                        dish_table_user = copy_dict(dish_on_table_per_user[index])
                    else:
                        dish_table_user = {}
                    dishes_toadd = copy_dict(particles[index].dish_on_table_todelete)
                    for user in dishes_toadd:
                        if user not in dish_table_user:
                            dish_table_user[user] = {}
                        for t in dishes_toadd[user]:
                            assert t not in dish_table_user[user]
                            dish_table_user[user][t] = dishes_toadd[user][t]
                    new_dish_on_table_per_user.append(dish_table_user)

                # delete history from new particles
                for index in new_particle_indices:
                    particles[index].table_history_with_user = []
                    for user in particles[index].dish_on_table_todelete:
                        particles[index].dish_on_table_todelete[user] = {}

                for index in new_particle_indices:
                    particles[index].table_history_with_user = []
                    new_particle = particles[index].copy()
                    new_particle.reseed(prng.randint(maxint))
                    new_particle.reset_weight()
                    new_particles.append(new_particle)
                    inferred_tables[new_particle.uid] = \
                        copy(inferred_tables[particles[index].uid])
                particles = new_particles
                table_history_with_user = new_table_history_with_user
                dish_on_table_per_user = new_dish_on_table_per_user

                # If inferred tables dictionary grows too big, prune it
                if len(inferred_tables) > 50 * params.num_particles:
                    new_inferred_tables = {}
                    for p in particles:
                        new_inferred_tables[p.uid] = copy(inferred_tables[p.uid])
                    del inferred_tables
                    inferred_tables = new_inferred_tables
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

    #print (len (particles))
    #print (final_particle_id)
    #for k in particles[(final_particle_id + 1) % len (particles)].per_topic_word_counts["docs"]:
    #    print(k, particles[(final_particle_id + 1) % len (particles)].per_topic_word_count_total["docs"][k], sum(particles[(final_particle_id + 1) % len (particles)].per_topic_word_counts["docs"][k].values()))

    #for k in particles[final_particle_id].per_topic_word_counts["docs"]:
    #    print(k, particles[final_particle_id].per_topic_word_count_total["docs"][k], sum(particles[final_particle_id].per_topic_word_counts["docs"][k].values()))

    table_history_with_user = table_history_with_user[final_particle_id]
    new_history = copy(final_particle.table_history_with_user)
    table_history_with_user.extend(new_history)
    final_particle.table_history_with_user = table_history_with_user
    dish_on_table_per_user = dish_on_table_per_user[final_particle_id]
    dishes_toadd = copy_dict(final_particle.dish_on_table_per_user)

    for user in dishes_toadd:
        if user not in dish_on_table_per_user:
            dish_on_table_per_user[user] = {}
        for t in dishes_toadd[user]:
            assert t not in dish_on_table_per_user[user]
            dish_on_table_per_user[user][t] = dishes_toadd[user][t]
    for user in final_particle.dish_on_table_todelete:
        if user not in dish_on_table_per_user:
            dish_on_table_per_user[user] = {}
        for t in final_particle.dish_on_table_todelete[user]:
            assert t not in dish_on_table_per_user[user]
            dish_on_table_per_user[user][t] = \
                final_particle.dish_on_table_todelete[user][t]
    final_particle.dish_on_table_per_user = dish_on_table_per_user

    final_particle.time_history_per_user = copy(time_history_per_user)
    final_particle.doc_history_per_user = copy(doc_history_per_user)
    final_particle.question_history_per_user = copy(question_history_per_user)
    final_particle.table_history_per_user = {}
    for (u_i, table) in final_particle.table_history_with_user:
        if u_i not in final_particle.table_history_per_user:
            final_particle.table_history_per_user[u_i] = []
        final_particle.table_history_per_user[u_i].append(table)
    final_particle.vocabulary = params.vocabulary
    # pool.close()
    with open(params.progress_file, mode='a') as temp:
        temp.write("Resampled %d times\n" % (count_resamples))
        temp.write("Finished in time: %.2f\n" %
                   (time() - start_tic))
    return final_particle, square_norms


def infer(history, docTypes, alpha_0, mu_0, omega=1, beta=1, theta_0=None,
          threads=1, num_particles=1,
          particle_weight_threshold=1, resample_every=10,
          update_kernels=True, mu_rate=0.6,
          # enable_log=False, logfile='particles.log',
          keep_alpha_history=False, progress_file=None, seed=None):
    """Runs the inference algorithm and returns a particle.

    Parameters
    ----------
    history : list
        A list of 4-tuples (user, time, content, metadata) that represents the
        event history that we want to infer our model on. Note that the content
        is itself a nested dictionary with keys as document types and values
        as documents themselves.

    docTypes: list
        A list of different document types whose content is present for each event.

    alpha_0 : tuple
        The Gamma prior parameter for a pattern's time kernel.

    mu_0 : tuple
        The Gamma prior parameter for the user activity rate.

    omega : float
        The time decay parameter.

    beta : float
        A parameter that controls the new-task probability.

    theta_0 :  dict, default is None
        If not None, theta_0 corresponds to the Dirichlet prior used for the
        word distribution for every document type. 
        It should have as many dimensions as the number of
        words. By default, this is the vector :math:`[1 / |V|, \ldots, 1 / |V|]`, where
        :math:`|V|` is the size of the vocabulary.

    threads : int, default is 1
        The number of CPU threads that will be used during inference.

    num_particles : int, default is 1
        The number of particles that the SMC algorithm will use.

    particle_weight_threshold : float, default is 1
        A parameter that controls when the particles need to be resampled

    resample_every : int, default is 10
        The frequency with which we check if we need to resample or not. The
        number is in inference steps (number of events)

    update_kernels : bool, default is True
        Controls wheter the time kernel parameter of each pattern will be
        updated from the posterior, or not.

    mu_rate : float, default is 0.6
        The learning-rate with which we update the activity rate of
        a user.

    keep_alpha_history : bool, default is False
        For debugging purpose, we may want to keep the complete history of the value
        of each pattern's time kernel parameter as we see more events in that
        pattern.

    progress_file : str, default is None
        Since the computation might be slow, we want to save progress
        information to a file instead of printing it. If None, a temporary,
        randomly-named file is generated for this purpose.

    seed: int, default is 512
        The seed to the random number generator.
    """

    vocabulary, users = _extract_words_users (history, docTypes)
    theta_0 = _initialize_document_distributions (vocabulary, docTypes, theta_0)

    if progress_file is None:
        with tempfile.NamedTemporaryFile(mode='a', suffix='.log', dir='.',
                                         delete=False) as temp:
            progress_file = temp.name
            print('Created temporary log file %s' % (progress_file))

    params = InferenceParameters(alpha_0, mu_0, omega, beta, theta_0, threads,
                                 num_particles, particle_weight_threshold,
                                 resample_every, update_kernels, mu_rate,
                                 keep_alpha_history, progress_file, seed,
                                 vocabulary, users)

    if threads == 1:
        return _infer_single_thread(history, params)
    else:
        raise NotImplementedError("Multi-threaded versoin not yet implemented")
