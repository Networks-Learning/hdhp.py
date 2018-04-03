import numpy as np
from hdhp import utils


def compute_pattern_overlap(p, vocab_types):
    """Computes the word overlap in the patterns of a process.


    Parameters
    ----------
    p : HDHProcess object


    Returns
    -------
    pattern_overlap : list
    """

    pattern_overlap = {vocab_type:[] for vocab_type in vocab_types}

    for vocab_type in vocab_types:

        words_per_pattern = {}
        for pattern in p.pattern_params[vocab_type]:
            words = set()
            for i in np.where(p.pattern_params[vocab_type][pattern] > 0)[0]:
                words.add(p.vocabulary[vocab_type][i])
            words_per_pattern[pattern] = words
        overlap = 0
        count = 0

        for i in words_per_pattern:
            for j in words_per_pattern:
                if i >= j:
                    continue
                sim = utils.word_overlap(words_per_pattern[i],
                                         words_per_pattern[j])
                overlap += sim
                pattern_overlap[vocab_type].append(sim)
                count += 1

        print ("Average overlap for " + vocab_type + " : ", str(overlap / count))
    return pattern_overlap
