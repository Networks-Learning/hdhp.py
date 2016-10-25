"""
    utils
    ~~~~~

    Provides helpful utilities for memoization and plotting.

    :copyright: 2016 Charalampos Mavroforakis, <cmav@bu.edu> and contributors.
    :license: ISC
"""
from __future__ import division

import calendar
import datetime
from copy import copy
from math import ceil

import seaborn as sns
from dateutil.relativedelta import relativedelta
from matplotlib.colors import hex2color, rgb2hex
from numpy import log as ln


class memoize:
    def __init__(self, function):
        self.function = function
        self.memoized = {}

    def __call__(self, *args):
        try:
            return self.memoized[args]
        except KeyError:
            self.memoized[args] = self.function(*args)
            return self.memoized[args]


def entropy(sets, N):
    res = 0
    for j in sets:
        res -= len(sets[j]) / N * ln(len(sets[j]) / N)
    return res


def copy_dict(original):
    new = {k: copy(original[k]) for k in original}
    return new


def qualitative_cmap(n_colors=17):
    """Returns a colormap suitable for a categorical plot with many categories.


    Parameters
    ----------
    n_colors : int, default is 17
        The number of colors that, usually, matches with the number of
        categories.


    Returns
    -------
    list
        A list of hex colors.
    """
    set1 = sns.mpl_palette("Set1", n_colors=9)
    hex_colors = [rgb2hex(rgb) for rgb in set1]
    hex_colors[5] = '#FFDE00'
    if n_colors <= 9:
        return hex_colors
    if n_colors <= 17:
        n_colors = 17
    else:
        n_colors = 8 * ceil((n_colors - 1) / 8)
    gradient = polylinear_gradient(hex_colors, n_colors)
    return gradient


def polylinear_gradient(colors, n):
    """Returns a list of colors forming linear gradients between
    all sequential pairs of colors. `n` specifies the total
    number of desired output colors.

    Code from http://bsou.io/posts/color-gradients-with-python
    """
    # The number of colors per individual linear gradient
    n_out = int(ceil(float(n - 1) / (len(colors) - 1))) + 1
    gradient = []
    for color_id in range(0, len(colors) - 1):
        color_subdivisions = linear_gradient(colors[color_id],
                                             colors[color_id + 1],
                                             n_out)
        gradient.extend(color_subdivisions[:-1])
    gradient.append(colors[-1])
    return gradient


def linear_gradient(start_hex, finish_hex="#FFFFFF", n=10):
    """Returns a gradient list of (n) colors between
    two hex colors. start_hex and finish_hex
    should be the full six-digit color string,
    including the number sign ("#FFFFFF")

    Code from http://bsou.io/posts/color-gradients-with-python
    """
    # Starting and ending colors in RGB form
    s = hex2color(start_hex)
    f = hex2color(finish_hex)
    # Initialize a list of the output colors with the starting color
    RGB_list = [rgb2hex(s)]
    # Calcuate a color at each evenly spaced value of t from 1 to n
    for t in range(1, n):
        # Interpolate RGB vector for color at the current value of t
        curr_vector = [s[j] + (t / (n - 1)) * (f[j] - s[j])
                       for j in range(3)]
        # Add it to our list of output colors
        RGB_list.append(rgb2hex(curr_vector))
    return RGB_list


def weighted_choice(weights, prng):
    """Samples from a discrete distribution.


    Parameters
    ----------
    weights : list
        A list of floats that identifies the distribution.

    prng : numpy.random.RandomState
        A pseudorandom number generator object.


    Returns
    -------
    int
    """
    rnd = prng.rand() * sum(weights)
    n = len(weights)
    i = -1
    while i < n - 1 and rnd >= 0:
        i += 1
        rnd -= weights[i]
    return i


def monthly_labels(t1, t2, every=6):
    """Returns labels for the months between two dates.

    The first label corresponds to the starting month, while the last label
    corresponds to the month after the ending month. This is done for better
    bracketing of the plot. By default, only the label every 6 months is
    non-empty.


    Parameters
    ----------
    t1 : datetime object
        Starting date

    t2 : datetime object
        End date

    every : int, default is 6


    Returns
    -------
    list
    """
    # Start from first day of month, for bracketing
    labels = []
    if t2.year > t1.year:
        # append months until the end of the year
        for month in range(t1.month, 13):
            labels.append(calendar.month_abbr[month][:3])
        for year in range(t1.year + 1, t2.year):
            for month in range(1, 13):
                labels.append(calendar.month_abbr[month][:3])
        for month in range(1, (t2.month % 12) + 1):
                labels.append(calendar.month_abbr[month][:3])
    else:
        for month in range(t1.month, (t2.month % 12) + 1):
                labels.append(calendar.month_abbr[month][:3])

    # append next month for bracketing
    labels.append(calendar.month_abbr[(t2.month % 12) + 1][:3])

    labels = [labels[i]
              if i % every == 0
              else ''
              for i in range(len(labels))]
    return labels


def daily_ticks(t1, t2):
    """Returns a list with the locations for the daily ticks on the timeline.


    Note
    ----
    The time scale is in days. The timeline is padded on the left with the
    required number of days so that it starts from the beginning of a month.
    Same happens to the right side, so  that it ends on a months end.


    Parameters
    ----------
    t1 : datetime object

    t2 : datetime object


    Returns
    -------
    list
    """
    start_of_timeline = datetime.datetime(year=t1.year, month=t1.month,
                                          day=1, hour=0, minute=0, second=0)
    end_of_timeline = datetime.datetime(year=t2.year, month=t2.month,
                                        day=calendar.monthrange(t2.year,
                                                                t2.month)[1],
                                        hour=23, minute=59, second=59)

    start_second = -int((t1 - start_of_timeline).total_seconds())
    end_second = int((end_of_timeline - t1).total_seconds())
    ticks = []
    for second in range(start_second, end_second + 86400, 86400):
        ticks.append(second / 86400)
    return ticks


def monthly_ticks_for_days(t1, t2):
    """Returns the ticks for the start of months in the timeline.


    Note
    ----
    The time scale is in days. The timeline has been padded on the left with
    the required number of days so that it starts from the beginning of a
    month. So, even though t1 stands for 0, the first actual tick might be
    at a negative location.
    Same happens to the right side, so  that it ends on a months end.


    Parameters
    ----------
    t1 : datetime object

    dt2 : datetime object


    Returns
    -------
    list : list of float
        List of tick locations indicating the start of a month.
    """
    monthly_ticks = []
    for tick in daily_ticks(t1, t2):
        current_date = t1 + datetime.timedelta(seconds=86400 * tick)
        if current_date.day == 1:
            monthly_ticks.append(tick)
    return monthly_ticks


def month_difference(t1, t2):
    """Computes the difference between two dates in months.


    Note
    ----
    This function is not symmetric. The decimal part of the result corresponds
    to the fraction of days of the last month before t2. If we call it with
    swapped arguments, result will differ in sign (as expected) but also in
    value.


    Parameters
    ----------
    t1 : datetime object

    t2 : datetime object


    Returns
    -------
    float
    """
    diff = 0
    relative = relativedelta(t2, t1)
    diff += relative.years * 12
    diff += relative.months
    future_date = t1 + relativedelta(months=diff)
    seconds_in_month = 86400 * calendar.monthrange(future_date.year,
                                                   future_date.month)[1]
    seconds_diff = relative.days * 86400 + relative.hours * 3600
    seconds_diff += relative.minutes * 60 + relative.seconds
    diff += seconds_diff / seconds_in_month
    return diff


def month_add(t1, months):
    """Adds a number of months to the given date.


    Note
    ----
    The decimal part of the value corresponds to the fraction of days of the
    last month that will be added.


    Parameters
    ----------
    t1 : datetime object

    months : float


    Returns
    -------
    t2 : datetime object
    """
    t2 = t1 + relativedelta(months=int(months // 1))
    days_in_month = calendar.monthrange(t2.year, t2.month)[1]
    t2 = t2 + datetime.timedelta(seconds=days_in_month * 86400 * (months % 1))
    return t2


def monthly_ticks_for_months(t1, t2):
    """Returns the ticks for the start of months in the timeline.


    Note
    ----
    The time scale is in months. The timeline has been padded on the left  so
    that it starts from the beginning of a month. So, even though t1 stands
    for 0, the first actual tick might be at a negative location.
    Same happens to the right side, so  that it ends on a months end.


    Parameters
    ----------
    t1 : datetime object

    dt2 : datetime object


    Returns
    -------
    list : list of float
        List of tick locations indicating the start of a month.
    """
    start_of_timeline = datetime.datetime(year=t1.year, month=t1.month,
                                          day=1, hour=0, minute=0, second=0)
    end_of_timeline = datetime.datetime(year=t2.year, month=t2.month,
                                        day=calendar.monthrange(t2.year,
                                                                t2.month)[1],
                                        hour=23, minute=59, second=59)
    end_of_timeline = end_of_timeline + datetime.timedelta(seconds=1)

    ticks = [-month_difference(start_of_timeline, t1)]
    current_tick_date = start_of_timeline
    next_tick_date = current_tick_date + relativedelta(months=1)
    while next_tick_date <= end_of_timeline:
        ticks.append(month_difference(t1, next_tick_date))
        current_tick_date = next_tick_date
        next_tick_date = current_tick_date + relativedelta(months=1)
    return ticks


def word_overlap(left_words, right_words):
    """Returns the Jaccard similarity between two sets.

    Note
    ----
    The topics are considered sets of words, and not distributions.

    Parameters
    ----------
    left_words : set
        The set of words for first topic

    right_words : set
        The set of words for the other topic


    Returns
    -------
    jaccard_similarity : float
    """
    intersection = len(left_words.intersection(right_words))
    union = len(left_words.union(right_words))
    jaccard = intersection / union
    return jaccard
