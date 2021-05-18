# coding = utf-8

import copy

import numpy as np
import pandas as pd

from .utils import computeStatistics, contained_in_other_keywords, count_words


def c_value(text,patterns):
    general_stats,_ = computeStatistics(patterns,text=text)
    for word in general_stats:
        count_in, sum_new_freq = contained_in_other_keywords(word,general_stats)
        substraction = general_stats[word]["new_freq"] - sum_new_freq
        if substraction >=0:
            general_stats[word]["new_freq"] = substraction
        general_stats[word]["sum_new_freq"] = sum_new_freq
        general_stats[word]["longertermwithword"] = count_in
        rank = np.log2(count_words(word)+1)
        word_freq = general_stats[word]["freq"]
        if count_in >0:
            if word_freq - (sum_new_freq/count_in) <=0:
                rank = rank * (word_freq-  (sum_new_freq/count_in) +0.5)
            else:
                rank = rank * (word_freq - (sum_new_freq/count_in))
        else:
            rank = rank * (word_freq)
        general_stats[word]["rank"] = rank
    return general_stats

def l_value(text,patterns):
    general_stats, _ = computeStatistics(patterns, text=text)
    for word in general_stats:
        count_in, sum_new_freq = contained_in_other_keywords(word, general_stats)
        substraction = general_stats[word]["new_freq"] - sum_new_freq
        if substraction >= 0:
            general_stats[word]["new_freq"] = substraction
        general_stats[word]["sum_new_freq"] = sum_new_freq
        general_stats[word]["longertermwithword"] = count_in
        rank = np.log2(count_words(word) + 1)
        word_freq = general_stats[word]["freq"]
        if count_in > 0:
            if word_freq - (sum_new_freq / count_in) <= 0:
                rank = rank * (word_freq - (sum_new_freq / count_in) + 0.5)
            else:
                rank = rank * (word_freq - (sum_new_freq / count_in))
        else:
            rank = rank * (word_freq)
        prob = general_stats[word]["pattern_freq"] / patterns.sum_all_patterns_frquency()
        rank = prob * rank
        general_stats[word]["rank"] = rank
    return general_stats

def lidf_value(corpus,patterns):
    general_stats, _ = computeStatistics(patterns, corpus=corpus)
    for word in general_stats:
        count_in, sum_new_freq = contained_in_other_keywords(word, general_stats)
        substraction = general_stats[word]["new_freq"] - sum_new_freq
        if substraction >= 0:
            general_stats[word]["new_freq"] = substraction
        general_stats[word]["sum_new_freq"] = sum_new_freq
        general_stats[word]["longertermwithword"] = count_in
        rank = np.log2(count_words(word) + 1)
        word_freq = general_stats[word]["freq"]
        if count_in > 0:
            if word_freq - (sum_new_freq / count_in) <= 0:
                rank = rank * (word_freq - (sum_new_freq / count_in) + 0.5)
            else:
                rank = rank * (word_freq - (sum_new_freq / count_in))
        else:
            rank = rank * (word_freq)
        prob = general_stats[word]["pattern_freq"] / patterns.sum_all_patterns_frquency()
        idf = np.log10(len(corpus)/general_stats[word]["num_doc"])
        rank = prob * rank * idf
        general_stats[word]["rank"] = rank
    return general_stats

def okapi(corpus,patterns,opt="AVG"):
    general_stats, stats_per_doc = computeStatistics(patterns, corpus=corpus)
    k1, b = 2., 0.5
    dl_avg = np.mean([len(stats_per_doc[d]) for d in stats_per_doc])
    for ix,doc in stats_per_doc.items():
        max_rank, min_rank = 0., 9999.0
        max_freq = max([doc[k]["freq"] for k in doc])
        len_doc = len(doc)
        for term,val in doc.items():
            norm_tf = val["freq"]/max_freq
            num_doc_gen = general_stats[term]["num_doc"]
            idf_bm25 = np.log10((len_doc - num_doc_gen +0.5 )/num_doc_gen+0.5)
            tf_bm25 = (norm_tf*(k1+1))/(norm_tf+k1*((1-b)+b*(len_doc/dl_avg)))
            rank = idf_bm25 *tf_bm25
            if np.isnan(rank):
                print(idf_bm25,tf_bm25)
            max_rank = max(max_rank,rank)
            min_rank = min(min_rank, rank)
            stats_per_doc[ix][term]["rank"] = rank
            stats_per_doc[ix][term]["rank_norm"] = rank/max_rank#(rank-min_rank)/(max_rank-min_rank)
    return apply_opt(general_stats,stats_per_doc,opt)

def tf_idf(corpus,patterns,opt="AVG"):
    general_stats, stats_per_doc = computeStatistics(patterns, corpus=corpus)
    for ix, doc in stats_per_doc.items():
        max_rank, min_rank = 0., 9999.0
        max_freq = max([doc[k]["freq"] for k in doc])
        len_doc = len(doc)
        for term, val in doc.items():
            norm_tf = val["freq"] / max_freq
            idf = np.log10((len_doc)/(general_stats[term]["num_doc"]+0.1))
            rank = norm_tf * idf
            max_rank = max(max_rank, rank)
            min_rank = min(min_rank, rank)
            stats_per_doc[ix][term]["rank"] = rank
            stats_per_doc[ix][term]["rank_norm"] = rank/max_rank#(rank - min_rank) / (max_rank - min_rank)
    return apply_opt(general_stats, stats_per_doc, opt)


def f_okapi_c(corpus,patterns,opt="AVG"):
    stats_okapi = okapi(corpus,patterns,opt)
    stats_c_value = c_value(pd.concat((corpus)),patterns)

    stats_final = copy.deepcopy(stats_okapi)
    for term in stats_okapi:
        rank_okapi = stats_okapi[term]["rank"]
        rank_c_value = stats_c_value[term]["rank"]
        stats_final[term]["rank"] = (2*(rank_okapi*rank_c_value))/(rank_okapi+rank_c_value)

    return stats_final

def f_tfidf_c(corpus,patterns,opt="AVG"):
    stats_tfidf = tf_idf(corpus,patterns,opt)
    stats_c_value = c_value(pd.concat((corpus)),patterns)

    stats_final = copy.deepcopy(stats_tfidf)
    for term in stats_tfidf:
        rank_tfidf = stats_tfidf[term]["rank"]
        rank_c_value = stats_c_value[term]["rank"]
        stats_final[term]["rank"] = (2*(rank_tfidf*rank_c_value))/(rank_tfidf+rank_c_value)

    return stats_final


def apply_opt(general_stats,stats_per_doc,opt):
    if not opt in ["MAX","SUM","AVG"]:
        raise ValueError("opt must be equal to MAX or SUM or AVG")
    for ix,doc in stats_per_doc.items():
        for term in doc:
            if not "rank" in general_stats[term]:
                general_stats[term]["rank"] = 0

            if opt == "MAX":
                general_stats[term]["rank"] = max(general_stats[term]["rank"],doc[term]["rank_norm"])
            if opt in ["SUM","AVG"]:
                general_stats[term]["rank"] +=  doc[term]["rank_norm"]
    if opt == "AVG":
        general_stats[term]["rank"] = general_stats[term]["rank"] / general_stats[term]["num_doc"]
    return general_stats
