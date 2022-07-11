# coding = utf-8
from biotex.pattern import Pattern
import re

def count_words(term):
    return len(term.split(" "))

def computeStatistics(pattern_instance : Pattern, text=None,corpus=None):
    """
    Return two dict, one that store overall corpus statistics of terms identified, the second store statistics
    of each term per document.

    Parameters
    ----------
    pattern_instance : Pattern
        pattern database
    text : str or None
    corpus : str or None

    Returns
    -------
    tuple(dict, dict)
        overalldocuments statistics dict, per document statistics dict
    """
    stats_general = {}
    stats_per_doc = {}
    longest_pat = pattern_instance.get_longest_pattern() + 1 # +1 to offset the double range()
    if not text is None:
        corpus = [text]
    for ix, doc in enumerate(corpus): # doc is a dataframe
        stats_per_doc[ix] = {}
        words = doc["word"].values
        partofspeech_vals = doc["pos"].values
        for iy,pos in enumerate(partofspeech_vals):
            for i in range(longest_pat):
                pos_seq = [partofspeech_vals[iy+dec] for dec in range(i) if iy+dec <len(partofspeech_vals)]
                flag,pattern,frequency = pattern_instance.match(pos_seq)
                if flag:
                    term = " ".join([words[iy+dec] for dec in range(i) if iy+dec <len(partofspeech_vals)]).lower()
                    if not term in stats_general:
                        stats_general[term] = {
                            "freq" : 1,
                            "new_freq": 0,
                            "num_doc":1,
                            "last_saw":ix,
                            "pattern_used":pattern,
                            "pattern_freq":frequency,
                            "num_words" :count_words(term)
                        }
                    elif term in stats_general:
                        stats_general[term]["freq"] += 1
                        if ix < stats_general[term]["last_saw"]:
                            stats_general[term]["last_saw"] = ix
                            stats_general[term]["num_doc"] += 1

                    if term not in stats_per_doc[ix]:
                        stats_per_doc[ix][term] = {
                            "freq":0,
                            "new_freq":0
                        }
                    stats_per_doc[ix][term]["freq"] +=1
    return stats_general, stats_per_doc

def term_in_term(term1,term2):
    """
    Return true if a term appears in a second one.

    Parameters
    ----------
    term1 : str
    term2 : str

    Returns
    -------
    bool
        if term1 is in term2
    """
    found =  re.findall("({0} | {0} | {0})".format(term1),term2)
    return len(found)>0

def contained_in_other_keywords(term,general_stats_dict):
    """
    Return the number of time a term appears in an other identified term and the sum of their frequency.

    Parameters
    ----------
    term : str
    general_stats_dict : dict

    Returns
    -------
    tuple(int, int)
        count of other terms in which the selectecr term appears, sum of these termes "new frequency"
    """
    count_in = 0
    sum_new_freq = 0
    data_keywords = [[k,v["num_words"],v["new_freq"]] for k,v in general_stats_dict.items()]
    for data in data_keywords:
        if term_in_term(term, data[0]):
            count_in += 1
            sum_new_freq += data[-1]
    return count_in,sum_new_freq
