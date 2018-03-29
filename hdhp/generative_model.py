"""
    generative_model
    ~~~~~~~~~~~~~~~~~

    Provides the following generative model:
    :class:`generative_model.HDHProcess`
      Implements the generative model of a hierarchical
      Dirichlet-Hawkes process.

    :copyright: 2016 Charalampos Mavroforakis, <cmav@bu.edu> and contributors.
    :license: ISC
"""
from __future__ import division, print_function

import datetime
from collections import defaultdict

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from numpy import log as ln
from numpy import asfortranarray, exp
from numpy.random import RandomState
from scipy.misc import logsumexp
from sklearn.utils import check_random_state
from utils import (qualitative_cmap, weighted_choice, monthly_labels,
                   monthly_ticks_for_days, monthly_ticks_for_months,
                   month_add)
from itertools import izip


class HDHProcess:
    def __init__(self, num_patterns, alpha_0, mu_0, vocabulary, vocab_types, total_user, cousers=None, omega=1,
                 doc_length={'docs': 20, 'auths': 20}, doc_min_length={'docs': 5, 'auths': 5},
                 words_per_pattern={'docs': 10, 'auths': 10},
                 random_state=None, generate=False):
        """
        Parameters
        ----------
        num_patterns : int
            The maximum number of patterns that will be shared across
            the users.

        alpha_0 : tuple
            The parameter that is used when sampling the time kernel weights
            of each pattern. The distribution that is being used is a Gamma.
            This tuple should be of the form (shape, scale).

        mu_0 : tuple
            The parameter of the Gamma distribution that is used to sample
            each user's \mu (activity level). This tuple should be of the
            form (shape, scale).

        vocabulary : list
            The list of available words to use when generating documents.

        omega : float, default is 1
            The decay parameter for the decay of the exponential decay kernel.

        doc_length : int, default is 20
            The maximum number of words per document.

        doc_min_length : int, default is 5
            The minimum number of words per document.

        words_per_pattern: int, default is 10
            The number of words that will have a non-zero probability to appear
            in each pattern.

        random_state: int or RandomState object, default is None
            The random number generator.
        """
        self.prng = check_random_state(random_state)
        self.doc_prng = RandomState(self.prng.randint(200000000))
        self.time_kernel_prng = RandomState(self.prng.randint(2000000000))
        self.pattern_param_prng = RandomState(self.prng.randint(2000000000))
        self.cousers_param_prng = RandomState(self.prng.randint(2000000000))

        self.num_patterns = num_patterns
        self.alpha_0 = alpha_0
        self.vocabulary = vocabulary
        self.mu_0 = mu_0
        self.document_length = doc_length
        self.document_min_length = doc_min_length
        self.omega = omega
        self.words_per_pattern = words_per_pattern
        self.vocab_types = vocab_types
        ######
        self.cousers = cousers
        self.total_users = total_user

        if generate:
            self.pattern_params = self.sample_pattern_params()
            self.time_kernels = self.sample_time_kernels()
            self.pattern_popularity = self.sample_pattern_popularity()
            self.couser_params = self.sample_couser_params()

        # Initialize all the counters etc.
        self.reset()

    def reset(self):
        """Removes all the events and users already sampled.


        Note
        ----
        It does not reseed the random number generator. It also retains the
        already sampled pattern parameters (word distributions and alphas)
        """
        self.mu_per_user = {i: self.sample_mu() for i in xrange(self.total_users)}
        self.num_users = 0
        self.time_history = []
        self.time_history_per_user = {}
        self.table_history_per_user = {}
        self.dish_on_table_per_user = {}
        self.dish_counters = defaultdict(int)
        self.last_event_user_pattern = defaultdict(lambda: defaultdict(int))
        self.total_tables = 0
        self.first_observed_time = {}
        self.user_table_cache = defaultdict(dict)
        self.table_history_per_user = defaultdict(list)
        self.time_history_per_user = defaultdict(list)
        self.document_history_per_user = defaultdict(list)
        self.dish_on_table_per_user = defaultdict(dict)
        self.cache_per_user = defaultdict(dict)
        self.total_tables_per_user = defaultdict(int)
        self.events = []
        self.per_pattern_word_counts = {vocab_type: defaultdict(lambda: defaultdict(int)) for vocab_type in
                                        self.vocab_types}
        self.per_pattern_word_count_total = {vocab_type: defaultdict(int) for vocab_type in self.vocab_types}
        #####
        self.user_dish_cache = defaultdict(dict)
        self.cousers_history_per_user = defaultdict(list)

    def sample_couser_params(self):
        """ Returns the co-authoring distributions for every user """
        couser_params = {}
        for user in xrange(self.total_users):
            # theta = [1.0 / self.num_users for i in range(self.num_users)]
            couser_params[user] = self.cousers_param_prng.dirichlet(self.cousers[user, :])

        return couser_params

    def sample_pattern_params(self):
        """Returns the word distributions for each pattern.


        Returns
        -------
        parameters : list
            A list of word distributions, one for each pattern.
        """
        sampled_params = {vocab_type: dict() for vocab_type in self.vocab_types}
        for vocab_type in self.vocab_types:
            V = len(self.vocabulary[vocab_type])

            for pattern in range(self.num_patterns):
                custom_theta = [0] * V
                words_in_pattern = self.prng.choice(V, size=self.words_per_pattern[vocab_type],
                                                    replace=False)
                for word in words_in_pattern:
                    custom_theta[word] = 100. / self.words_per_pattern[vocab_type]
                sampled_params[vocab_type][pattern] = \
                    self.pattern_param_prng.dirichlet(custom_theta)
        return sampled_params

    def sample_time_kernels(self):
        """Returns the time decay parameter of each pattern.


        Returns
        -------
        alphas : list
            A list of time decay parameters, one for each pattern.
        """
        alphas = {pattern: self.time_kernel_prng.gamma(self.alpha_0[0],
                                                       self.alpha_0[1])
                  for pattern in range(self.num_patterns)}
        return alphas

    def sample_pattern_popularity(self):
        """Returns a popularity distribution over the patterns.


        Returns
        -------
        pattern_popularities : list
            A list with the popularity distribution of each pattern.
        """
        pattern_popularity = {}
        pi = self.pattern_param_prng.dirichlet([1 for pattern
                                                in range(self.num_patterns)])
        for pattern_i, pi_i in enumerate(pi):
            pattern_popularity[pattern_i] = pi_i
        return pattern_popularity

    def sample_mu(self):
        """Samples a value from the prior of the base intensity mu.


        Returns
        -------
        mu_u : float
            The base intensity of a user, sampled from the prior.
        """
        return self.prng.gamma(self.mu_0[0], self.mu_0[1])

    def sample_cousers_for(self, user):
        counts = self.prng.multinomial(self.total_users, self.couser_params[user])
        others = list()
        for i, count in enumerate(counts):
            if count > 0 and not i == user:
                others.append(i)
        return others

    def sample_next_time(self, pattern, user, reset=True):
        """Samples the time of the next event of a pattern for a given user.


        Parameters
        ----------
        pattern : int
            The pattern index that we want to sample the next event for.

        user : int
            The index of the user that we want to sample for.


        Returns
        -------
        timestamp : float
        """
        U = self.prng.rand
        mu_u = self.mu_per_user[user]
        lambda_u_pattern = mu_u * self.pattern_popularity[pattern]

        if user not in self.user_dish_cache or pattern not in self.user_dish_cache[user]:
            lambda_star = lambda_u_pattern
            s = -1 / lambda_star * np.log(U())
            return s
        else:

            # Add the \alpha of the most recent table (previous event) in the
            # user-pattern intensity
            s = self.last_event_user_pattern[user][pattern]
            alpha = self.time_kernels[pattern]

            t_last, sum_kernels = self.user_dish_cache[user][pattern]
            update_value = self.kernel(s, t_last)

            # update_value should be 1, so practically we are just adding
            # \alpha to the intensity dt after the event
            pattern_intensity = alpha * sum_kernels * update_value
            pattern_intensity += alpha * update_value
            lambda_star = lambda_u_pattern + pattern_intensity

            # New event
            accepted = False
            while not accepted:
                s = s - 1 / lambda_star * np.log(U())
                # Rejection test
                t_last, sum_kernels = self.user_dish_cache[user][pattern]
                update_value = self.kernel(s, t_last)
                # update_value should be 1, so practically we are just adding
                # \alpha to the intensity dt after the event
                pattern_intensity = alpha * sum_kernels * update_value
                pattern_intensity += alpha * update_value

                lambda_s = lambda_u_pattern + pattern_intensity
                if U() < lambda_s / lambda_star:
                    return s
                else:
                    lambda_star = lambda_s

    def sample_user_events(self, min_num_events=100, max_num_events=None,
                           t_max=None):
        """Samples events for a user.


        Parameters
        ----------
        min_num_events : int, default is 100
            The minimum number of events to sample.

        max_num_events : int, default is None
            If not None, this is the maximum number of events to sample.

        t_max : float, default is None
            The time limit until which to sample events.


        Returns
        -------
        events : list
            A list of the form [(t_i, doc_i, user_i, meta_i), ...] sorted by
            increasing time that has all the events of the sampled users.
            Above, doc_i is the document and meta_i is any sort of metadata
            that we want for doc_i, e.g. question_id. The generator will return
            an empty list for meta_i.
        """

        next_time_per_pattern = {(pattern, user): self.sample_next_time(pattern, user) for pattern in
                                 xrange(self.num_patterns) for user in xrange(self.total_users)}
        iteration = 0
        over_tmax = False

        while iteration < min_num_events or not over_tmax:
            if max_num_events is not None and iteration > max_num_events:
                break
            z_n, user = min(next_time_per_pattern, key=next_time_per_pattern.get)
            cousers = self.sample_cousers_for(user)
            t_n = next_time_per_pattern[(z_n, user)]

            if t_max is not None and t_n > t_max:
                over_tmax = True
                break

            num_tables_user = self.total_tables_per_user[user] \
                if user in self.total_tables_per_user else 0
            tables = range(num_tables_user)
            tables = [table for table in tables
                      if self.dish_on_table_per_user[user][table] == z_n]
            intensities = []

            alpha = self.time_kernels[z_n]
            for table in tables:
                t_last, sum_kernels = self.user_table_cache[user][table]
                update_value = self.kernel(t_n, t_last)
                table_intensity = alpha * sum_kernels * update_value
                table_intensity += alpha * update_value
                intensities.append(table_intensity)
            intensities.append(self.mu_per_user[user] * self.pattern_popularity[z_n])
            log_intensities = [ln(inten_i)
                               if inten_i > 0 else -float('inf')
                               for inten_i in intensities]

            normalizing_log_intensity = logsumexp(log_intensities)
            intensities = [exp(log_intensity - normalizing_log_intensity)
                           for log_intensity in log_intensities]
            k = weighted_choice(intensities, self.prng)

            if k < len(tables):
                # Assign to already existing table
                table = tables[k]
                # update cache for that table
                t_last, sum_kernels = self.user_table_cache[user][table]
                update_value = self.kernel(t_n, t_last)
                sum_kernels += 1
                sum_kernels *= update_value
                self.user_table_cache[user][table] = (t_n, sum_kernels)
            else:
                table = num_tables_user
                self.total_tables += 1
                self.total_tables_per_user[user] += 1
                # Since this is a new table, initialize the cache accordingly
                self.user_table_cache[user][table] = (t_n, 0)
                self.dish_on_table_per_user[user][table] = z_n

            if z_n not in self.first_observed_time or \
                            t_n < self.first_observed_time[z_n]:
                self.first_observed_time[z_n] = t_n
            self.dish_counters[z_n] += 1

            doc_n = self.sample_document(z_n)
            self._update_word_counters(doc_n, z_n)
            self.document_history_per_user[user] \
                .append(doc_n)
            self.table_history_per_user[user].append(table)
            self.time_history_per_user[user].append(t_n)
            self.cousers_history_per_user[user].append(cousers)
            self.last_event_user_pattern[user][z_n] = t_n

            #####
            if user not in self.user_dish_cache:
                self.user_dish_cache[user] = {}
            if z_n not in self.user_dish_cache[user]:
                self.user_dish_cache[user][z_n] = (t_n, 0)
            else:
                t_last, sum_kernels = self.user_dish_cache[user][z_n]
                update_value = self.kernel(t_n, t_last)
                sum_kernels += 1
                sum_kernels *= update_value
                self.user_dish_cache[user][z_n] = (t_last, sum_kernels)

            for u in cousers:
                self.last_event_user_pattern[u][z_n] = t_n

                if u not in self.user_dish_cache:
                    self.user_dish_cache[u] = {}
                if z_n not in self.user_dish_cache[u]:
                    self.user_dish_cache[u][z_n] = (t_n, 0)
                else:
                    t_last, sum_kernels = self.user_dish_cache[u][z_n]
                    update_value = self.kernel(t_n, t_last)
                    sum_kernels += 1
                    sum_kernels *= update_value
                    self.user_dish_cache[u][z_n] = (t_n, sum_kernels)

            # Resample time for that pattern
            next_time_per_pattern[(z_n, user)] = self.sample_next_time(z_n, user)
            for u in cousers:
                next_time_per_pattern[(z_n, u)] = self.sample_next_time(z_n, u)
            iteration += 1

        # events = [(self.time_history_per_user[user][i],
        #            self.document_history_per_user[user][i], [user] + self.cousers_history_per_user[user][i], [])
        #           for i in range(len(self.time_history_per_user[user]))]

        # Update the full history of events with the ones generated for the
        # current user and re-order everything so that the events are
        # ordered by their timestamp
        for user in xrange(self.total_users):
            events = [(self.time_history_per_user[user][i],
                       self.document_history_per_user[user][i], [user] + self.cousers_history_per_user[user][i],
                       [self.dish_on_table_per_user[user][self.table_history_per_user[user][i]]])
                      for i in xrange(len(self.time_history_per_user[user]))]
            self.events.extend(events)

            # Update the full history of events with the ones generated for the
            # current user and re-order everything so that the events are
            # ordered by their timestamp
        self.events = sorted(self.events, key=lambda x: x[0])
        return self.events

    def kernel(self, t_i, t_j):
        """Returns the kernel function for t_i and t_j.


        Parameters
        ----------
        t_i : float
            Timestamp representing `now`.

        t_j : float
            Timestamp representaing `past`.


        Returns
        -------
        float
        """
        return exp(-self.omega * (t_i - t_j))

    def sample_document(self, pattern):
        """Sample a random document from a specific pattern.


        Parameters
        ----------
        pattern : int
            The pattern from which to sample the content.


        Returns
        -------
        str
            A space separeted string that contains all the sampled words.
        """
        sampled_doc = {vocab_type: ' ' for vocab_type in self.vocab_types}

        for vocab_type in sampled_doc:
            length = self.doc_prng.randint(self.document_length[vocab_type]) + \
                     self.document_min_length[vocab_type]
            words = self.doc_prng.multinomial(length, self.pattern_params[vocab_type][pattern])

            sampled_doc[vocab_type] = ' '.join([self.vocabulary[vocab_type][i]
                                                for i, repeats in enumerate(words)
                                                for j in range(repeats)])

        return sampled_doc

    def _update_word_counters(self, doc, pattern):
        """Updates the word counters of the process for the particular document


        Parameters
        ----------
        doc : list
            A list with the words in the document.

        pattern : int
            The index of the latent pattern that this document belongs to.
        """
        for vocab_type in doc:
            for word in doc[vocab_type].split():
                self.per_pattern_word_counts[vocab_type][pattern][word] += 1
                self.per_pattern_word_count_total[vocab_type][pattern] += 1
        return

    def pattern_content_str(self, patterns=None, show_words=-1,
                            min_word_occurence=5):
        """Return the content information for the patterns of the process.


        Parameters
        ----------
        patterns : list, default is None
            If this list is provided, only information about the patterns in
            the list will be returned.

        show_words : int, default is -1
            The maximum number of words to show for each pattern. Notice that
            the words are sorted according to their occurence count.

        min_word_occurence : int, default is 5
            Only show words that show up at least `min_word_occurence` number
            of times in the documents of the respective pattern.


        Returns
        -------
        str
            A string with all the content information
        """
        if patterns is None:
            patterns = self.per_pattern_word_counts.keys()
        text = ['___Pattern %d___ \n%s\n%s'
                % (pattern,
                   '\n'.join(['%s : %d' % (k, v)
                              for i, (k, v)
                              in enumerate(sorted(self.per_pattern_word_counts[pattern].iteritems(),
                                                  key=lambda x: (x[1], x[0]),
                                                  reverse=True))
                              if v >= min_word_occurence
                              and (show_words == -1
                                   or (show_words > 0 and i < show_words))]
                             ),
                   ' '.join([k
                             for i, (k, v)
                             in enumerate(sorted(self.per_pattern_word_counts[pattern].iteritems(),
                                                 key=lambda x: (x[1], x[0]),
                                                 reverse=True))
                             if v < min_word_occurence
                             and (show_words == -1
                                  or (show_words > 0 and i < show_words))])
                   )
                for pattern in self.per_pattern_word_counts
                if pattern in patterns]
        return '\n\n'.join(text)

    def user_patterns_set(self, user):
        """Return the patterns that a specific user adopted.


        Parameters
        ----------
        user : int
            The index of a user.


        Returns
        -------
        set
            The set of the patterns that the user adopted.
        """
        pattern_list = [self.dish_on_table_per_user[user][table]
                        for table in self.table_history_per_user[user]]
        return list(set(pattern_list))

    def user_pattern_history_str(self, user=None, patterns=None,
                                 show_time=True, t_min=0):
        """Returns a representation of the history of a user's actions and the
        pattern that they correspond to.


        Parameters
        ----------
        user : int, default is None
            An index to the user we want to inspect. If None, the function
            runs over all the users.

        patterns : list, default is None
            If not None, limit the actions returned to the ones that belong in
            the provided patterns.

        show_time : bool, default is True
            Control wether the timestamp will appear in the representation or
            not.

        t_min : float, default is 0
            The timestamp after which we only consider actions.


        Returns
        -------
        str
        """
        if patterns is not None and type(patterns) is not set:
            patterns = set(patterns)

        return '\n'.join(['%spattern=%2d task=%3d (u=%d)  %s' %
                          ('%5.3g ' % t if show_time else '', dish, table,
                           u, doc)
                          for u in range(self.num_users)
                          for ((t, doc), (table, dish)) in
                          zip([(t, d)
                               for t, d in zip(self.time_history_per_user[u],
                                               self.document_history_per_user[u])],
                              [(table, self.dish_on_table_per_user[u][table])
                               for table in self.table_history_per_user[u]])
                          if (user is None or user == u)
                          and (patterns is None or dish in patterns)
                          and t >= t_min])

    def _plot_user(self, user, fig, num_samples, T_max, task_detail,
                   seed=None, patterns=None, colormap=None, T_min=0, paper=True):
        """Helper function that plots.
        """
        tics = np.arange(T_min, T_max, (T_max - T_min) / num_samples)
        tables = sorted(set(self.table_history_per_user[user]))
        active_tables = set()
        dish_set = set()

        times = [t for t in self.time_history_per_user[user]]
        start_event = min([i for i, t
                           in enumerate(self.time_history_per_user[user])
                           if t >= T_min])
        table_cache = {}  # partial sum for each table
        dish_cache = {}  # partial sum for each dish
        table_intensities = [[] for _ in range(len(tables))]
        dish_intensities = [[] for _ in range(len(self.time_kernels))]
        i = 0  # index of tics
        j = start_event  # index of events

        while i < len(tics):
            if j >= len(times) or tics[i] < times[j]:
                # We need to measure the intensity
                dish_intensities, table_intensities = \
                    self._measure_intensities(tics[i], dish_cache=dish_cache,
                                              table_cache=table_cache,
                                              tables=tables,
                                              user=user,
                                              dish_intensities=dish_intensities,
                                              table_intensities=table_intensities)
                i += 1
            else:
                # We need to record an event and update our caches
                dish_cache, table_cache, active_tables, dish_set = \
                    self._update_cache(times[j],
                                       dish_cache=dish_cache,
                                       table_cache=table_cache,
                                       event_table=self.table_history_per_user[user][j],
                                       tables=tables,
                                       user=user,
                                       active_tables=active_tables,
                                       dish_set=dish_set)
                j += 1
        if patterns is not None:
            dish_set = set([d for d in patterns if d in dish_set])
        dish_dict = {dish: i for i, dish in enumerate(dish_set)}

        if colormap is None:
            prng = RandomState(seed)
            num_dishes = len(dish_set)
            colormap = qualitative_cmap(n_colors=num_dishes)
            colormap = prng.permutation(colormap)

        if not task_detail:
            for dish in dish_set:
                fig.plot(tics, dish_intensities[dish],
                         color=colormap[dish_dict[dish]], linestyle='-',
                         label="Pattern " + str(dish),
                         linewidth=3.)
        else:
            for table in active_tables:
                dish = self.dish_on_table_per_user[user][table]
                if patterns is not None and dish not in patterns:
                    continue
                fig.plot(tics, table_intensities[table],
                         color=colormap[dish_dict[dish]],
                         linewidth=3.)
            for dish in dish_set:
                if patterns is not None and dish not in patterns:
                    continue
                fig.plot([], [],
                         color=colormap[dish_dict[dish]], linestyle='-',
                         label="Pattern " + str(dish),
                         linewidth=5.)
        return fig

    def _measure_intensities(self, t, dish_cache, table_cache, tables, user,
                             dish_intensities, table_intensities):
        """Measures the intensities of all tables and dishes for the plot func.


        Parameters
        ----------
        t : float

        dish_cache : dict

        table_cache : dict

        tables : list

        user : int

        dish_intensities : dict

        table_intensities : dict


        Returns
        -------
        dish_intensities : dict

        table_intensities : dict
        """
        updated_dish = [False] * len(self.time_kernels)
        for table in tables:
            dish = self.dish_on_table_per_user[user][table]
            lambda_uz = self.mu_per_user[user] * self.pattern_popularity[dish]
            alpha = self.time_kernels[dish]
            if dish in dish_cache:
                t_last_dish, sum_kernels_dish = dish_cache[dish]
                update_value_dish = self.kernel(t, t_last_dish)
                dish_intensity = lambda_uz + alpha * sum_kernels_dish * \
                                             update_value_dish
                dish_intensity += alpha * update_value_dish
            else:
                dish_intensity = lambda_uz
            if table in table_cache:
                # table already exists
                t_last_table, sum_kernels_table = table_cache[table]
                update_value_table = self.kernel(t, t_last_table)
                table_intensity = alpha * sum_kernels_table * \
                                  update_value_table
                table_intensity += alpha * update_value_table
            else:
                # table does not exist yet
                table_intensity = 0
            table_intensities[table].append(table_intensity)
            if not updated_dish[dish]:
                # make sure to update dish only once for all the tables
                dish_intensities[dish].append(dish_intensity)
                updated_dish[dish] = True
        return dish_intensities, table_intensities

    def _update_cache(self, t, dish_cache, table_cache, event_table, tables,
                      user, active_tables, dish_set):
        """Updates the caches for the plot function when an event is recorded.


        Parameters
        ----------
        t : float

        dish_cache : dict

        table_cache : dict

        event_table : int

        tables : list

        user : int

        active_tables : set

        dish_set : set


        Returns
        -------
        dish_cache : dict

        table_cache : dict

        active_tables : set

        dish_set : set
        """
        dish = self.dish_on_table_per_user[user][event_table]
        active_tables.add(event_table)
        dish_set.add(dish)
        if event_table not in table_cache:
            table_cache[event_table] = (t, 0)
        else:
            t_last, sum_kernels = table_cache[event_table]
            update_value = self.kernel(t, t_last)
            sum_kernels += 1
            sum_kernels *= update_value
            table_cache[event_table] = (t, sum_kernels)
        if dish not in dish_cache:
            dish_cache[dish] = (t, 0)
        else:
            t_last, sum_kernels = dish_cache[dish]
            update_value = self.kernel(t, t_last)
            sum_kernels += 1
            sum_kernels *= update_value
            dish_cache[dish] = (t, sum_kernels)
        return dish_cache, table_cache, active_tables, dish_set

    def plot(self, num_samples=500, T_min=0, T_max=None, start_date=None,
             users=None, user_limit=50, patterns=None, task_detail=False,
             save_to_file=False, filename="user_timelines",
             intensity_threshold=None, paper=True, colors=None, fig_width=20,
             fig_height_per_user=5, time_unit='months', label_every=3,
             seed=None):
        """Plots the intensity of a set of users for a set of patterns over a
        time period.

        In this plot, each user is a separate subplot and for each user the
        plot shows her event_rate for each separate pattern that she has been
        active at.


        Parameters
        ----------
        num_samples : int, default is 500
            The granularity level of the intensity line. Smaller number of
            samples results in faster plotting, while larger numbers give
            much more detailed result.

        T_min : float, default is 0
            The minimum timestamp that the plot shows, in seconds.

        T_max : float, default is None
            If not None, this is the maximum timestamp that the plot considers,
            in seconds.

        start_date : datetime, default is None
            If provided, this is the actual datetime that corresponds to
            time 0. This is required if `paper` is True.

        users : list, default is None
            If provided, this list contains the id's of the users that will be
            plotted. Actually, only the first `user_limit` of them will be
            shown.

        user_limit : int, default is 50
            The maximum number of users to plot.

        patterns : list, default is None
            The list of patterns that will be shown in the final plot. If None,
            all of the patterns will be plotted.

        task_detail : bool, default is False
            If True, thee plot has one line per task. Otherwise, we only plot
            the commulative intensity of all tasks under the same pattern.

        save_to_file : bool, default is False
            If True, the plot will be saved to a `pdf` and a `png` file.

        filename : str, default is 'user_timelines'
            The name of the output file that will be used when saving the plot.

        intensity_threshold : float, default is None
            If provided, this is the maximum intensity value that will be
            plotted, i.e. the y_max that will be the cut-off threshold for the
            y-axis.

        paper : bool, default is True
            If True, the plot result will be the same as the figures that are
            in the published paper.

        colors : list, default is None
            A list of colors that will be used for the plot. Each color will
            correspond to a single pattern, and will be shared across all the
            users.

        fig_width : int, default is 20
            The width of the figure that will be returned.

        fig_height_per_user : int, default is 5
            The height of each separate user-plot of the final figure. If
            multiplied by the number of users, this determines the total height
            of the figure. Notice that due to a matplotlib constraint(?) the
            total height of  the figure cannot be over 70.

        time_unit : str, default is 'months'
            Controls wether the time units is measured in days (in
            which case it should be set to 'days') or months.

        label_every : int, default is 3
            The frequency of the labels that show in the x-axis.

        seed : int, default is None
            A seed to the random number generator used to assign colors to
            patterns.


        Returns
        -------
        fig : matplotlib.Figure object
        """
        prng = RandomState(seed)
        num_users = len(self.dish_on_table_per_user)
        if users is None:
            users = range(num_users)
        num_users_to_plot = min(len(users), user_limit)
        users = users[:num_users_to_plot]
        if T_max is None:
            T_max = max([self.time_history_per_user[user][-1]
                         for user in users])
        fig = plt.figure(figsize=(fig_width,
                                  min(fig_height_per_user * num_users_to_plot,
                                      70)))

        num_patterns_global = len(self.time_kernels)
        colormap = qualitative_cmap(n_colors=num_patterns_global)
        colormap = prng.permutation(colormap)
        if colors is not None:
            colormap = matplotlib.colors.ListedColormap(colors)
        if paper:
            sns.set_style('white')
            sns.despine(bottom=True, top=False, right=False, left=False)

        user_plt_axes = []
        max_intensity = -float('inf')
        for i, user in enumerate(users):
            if user not in self.time_history_per_user \
                    or not self.time_history_per_user[user] \
                    or self.time_history_per_user[user][-1] < T_min:
                # no events generated for this user during the time window
                # we are interested in
                continue
            if patterns is not None:
                user_patterns = set([self.dish_on_table_per_user[user][table]
                                     for table
                                     in self.table_history_per_user[user]])
                if not any([pattern in patterns for pattern in user_patterns]):
                    # user did not generate events in the patterns of interest
                    continue
            user_plt = plt.subplot(num_users_to_plot, 1, i + 1)
            user_plt = self._plot_user(user, user_plt, num_samples, T_max,
                                       task_detail=task_detail, seed=seed,
                                       patterns=patterns,
                                       colormap=colormap, T_min=T_min,
                                       paper=paper)
            user_plt.set_xlim((T_min, T_max))
            if paper:
                if start_date is None:
                    raise ValueError(
                        'For paper-level quality plots, the actual datetime for t=0 must be provided as `start_date`')
                if start_date.microsecond > 500000:
                    start_date = start_date.replace(microsecond=0) \
                                 + datetime.timedelta(seconds=1)
                else:
                    start_date = start_date.replace(microsecond=0)
                if time_unit == 'days':
                    t_min_seconds = T_min * 86400
                    t_max_seconds = T_max * 86400
                    t1 = start_date + datetime.timedelta(0, t_min_seconds)
                    t2 = start_date + datetime.timedelta(0, t_max_seconds)
                    ticks = monthly_ticks_for_days(t1, t2)
                    labels = monthly_labels(t1, t2, every=label_every)
                elif time_unit == 'months':
                    t1 = month_add(start_date, T_min)
                    t2 = month_add(start_date, T_max)
                    ticks = monthly_ticks_for_months(t1, t2)
                    labels = monthly_labels(t1, t2, every=label_every)
                labels[-1] = ''
                user_plt.set_xlim((ticks[0], ticks[-1]))
                user_plt.yaxis.set_ticks([])
                user_plt.xaxis.set_ticks(ticks)
                plt.setp(user_plt.xaxis.get_majorticklabels(), rotation=-0)
                user_plt.tick_params('x', length=10, which='major',
                                     direction='out', top=False, bottom=True)
                user_plt.xaxis.set_ticklabels(labels, fontsize=30)
                user_plt.tick_params(axis='x', which='major', pad=10)
                user_plt.get_xaxis().majorTicks[1].label1.set_horizontalalignment('left')
                for tick in user_plt.xaxis.get_major_ticks():
                    tick.label1.set_horizontalalignment('left')

            current_intensity = user_plt.get_ylim()[1]
            if current_intensity > max_intensity:
                max_intensity = current_intensity
            user_plt_axes.append(user_plt)
            if not paper:
                plt.title('User %2d' % user)
                plt.legend()
        if intensity_threshold is None:
            intensity_threshold = max_intensity
        for ax in user_plt_axes:
            ax.set_ylim((-0.2, intensity_threshold))
        if paper and save_to_file:
            print("Create image %s.png" % (filename))
            plt.savefig(filename + '.png', transparent=True)
            plt.savefig(filename + '.pdf', transparent=True)
            sns.set_style(None)
        return fig

    def user_patterns(self, user):
        """Returns a list with the patterns that a user has adopted.

        Parameters
        ----------
        user : int
        """
        pattern_list = [self.dish_on_table_per_user[user][table]
                        for table in self.table_history_per_user[user]]
        return list(set(pattern_list))

    def show_annotated_events(self, user=None, patterns=None, show_time=True,
                              T_min=0, T_max=None):
        """Returns a string where each event is annotated with the inferred
        pattern.


        Parameters
        ----------
        user : int, default is None
            If given, the events returned are limited to the selected user

        patterns : list, default is None
            If not None, an event is return only if it belongs to one of the
            selected patterns

        show_time : bool, default is True
            Controls whether the time of the event will be shown

        T_min : float, default is 0
            Controls the minimum timestamp after which the events will be shown

        T_max : float, default is None
            If given, T_max controls the maximum timestamp shown


        Returns
        -------
        str
        """
        if patterns is not None and type(patterns) is not set:
            patterns = set(patterns)

        if show_time:
            return '\n'.join(['%5.3g pattern=%3d task=%3d (u=%d)  %s' %
                              (t, dish, table, u, doc)
                              for u in range(self.num_users)
                              for ((t, doc), (table, dish)) in
                              zip([(t, d)
                                   for t, d in zip(self.time_history_per_user[u],
                                                   self.document_history_per_user[u])],
                                  [(table, self.dish_on_table_per_user[u][table])
                                   for table in self.table_history_per_user[u]])
                              if (user is None or user == u) and
                              (patterns is None or dish in patterns)
                              and t >= T_min
                              and (T_max is None or (T_max is not None and t <= T_max))])
        else:
            return '\n'.join(['pattern=%3d task=%3d (u=%d)  %s'
                              % (dish, table, u, doc)
                              for u in range(self.num_users)
                              for ((t, doc), (table, dish)) in
                              zip([(t, d)
                                   for t, d in zip(self.time_history_per_user[u],
                                                   self.document_history_per_user[u])],
                                  [(table, self.dish_on_table_per_user[u][table])
                                   for table in self.table_history_per_user[u]])
                              if (user is None or user == u) and
                              (patterns is None or dish in patterns)
                              and t >= T_min
                              and (T_max is None or (T_max is not None and t <= T_max))])

    def show_pattern_content(self, patterns=None, words=0, detail_threshold=5):
        """Shows the content distrubution of the inferred patterns.


        Parameters
        ----------
        patterns : list, default is None
            If not None, only the content of the selected patterns will be
            shown

        words : int, default is 0
            A positive number that control how many words will be shown.
            The words are being shown sorted by their likelihood, starting
            with the most probable.

        detail_threshold : int, default is 5
            A positive number that sets the lower bound in the number of times
            that a word appeared in a pattern so that its count is shown.


        Returns
        -------
        str
        """
        if patterns is None:
            patterns = self.per_pattern_word_count.keys()
        text = ['___Pattern %d___ \n%s\n%s'
                % (pattern,
                   '\n'.join(['%s : %d'
                              % (k, v)
                              for i, (k, v)
                              in enumerate(sorted(self.per_pattern_word_counts[pattern].iteritems(),
                                                  key=lambda x: (x[1], x[0]),
                                                  reverse=True))
                              if v >= detail_threshold and (words == 0 or i < words)]
                             ),
                   ' '.join([k for i, (k, v)
                             in enumerate(sorted(self.per_pattern_word_counts[pattern].iteritems(),
                                                 key=lambda x: (x[1], x[0]),
                                                 reverse=True))
                             if v < detail_threshold and (words == 0 or i < words)])
                   )
                for pattern in self.per_pattern_word_counts if pattern in patterns]
        return '\n\n'.join(text)

    def annotatedEventsIter(self, keep_sorted=True):

        events = [(t, dish, table, u, doc)
                  for u in xrange(self.total_users)
                  for ((t, doc), (table, dish)) in izip([(t, d)
                                                         for t, d in izip(self.time_history_per_user[u],
                                                                          self.document_history_per_user[u])],
                                                        [(table, self.dish_on_table_per_user[u][table])
                                                         for table in self.table_history_per_user[u]])]

        if keep_sorted: events = sorted(events, key=lambda x: x[0])

        for event in events:
            yield event
