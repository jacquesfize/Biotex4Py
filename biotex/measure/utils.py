# coding = utf-8
from biotex.pattern import Pattern
import re
import logging

def count_words(term):
    return len(term.split(" "))

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
    pattern = re.compile("({0} | {0} | {0})".format(term1))
    found =  pattern.search(term2)
    return found

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

    for k,v in general_stats_dict.items():
        if term_in_term(term, k):
            count_in += 1
            sum_new_freq += v["new_freq"]
    return count_in,sum_new_freq

def debug(func):
    def new_func(*args,**params):
        logging.debug(f"Compute measure {func.__name__}")
        res = func(*args,**params)
        logging.debug(f"{func.__name__} is done !")
        return res
    return new_func