import numpy as np
from hdhp import utils


def compute_pattern_overlap(p):
    """Computes the word overlap in the patterns of a process.


    Parameters
    ----------
    p : HDHProcess object


    Returns
    -------
    pattern_overlap : list
    """
    words_per_pattern = {}
    for pattern in p.pattern_params:
        words = set()
        for i in np.where(p.pattern_params[pattern] > 0)[0]:
            words.add(p.vocabulary[i])
        words_per_pattern[pattern] = words
    overlap = 0
    count = 0
    pattern_overlap = []
    for i in words_per_pattern:
        for j in words_per_pattern:
            if i >= j:
                continue
            sim = utils.word_overlap(words_per_pattern[i],
                                     words_per_pattern[j])
            overlap += sim
            pattern_overlap.append(sim)
            count += 1
    print "Average overlap:", overlap / count
    return pattern_overlap
